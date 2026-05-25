"""Phase 0b benchmark harness.

Compares HDR fusion methods on bracket sets. Writes per-room HTML reports
plus a summary CSV. Cannot validate without real brackets.
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps, ExifTags
from skimage import color


JPEG_LIKE = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
RAW_EXTS = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf"}
SUPPORTED_EXTS = JPEG_LIKE | RAW_EXTS
# Folders that may contain `brackets/` subdirs but aren't real dataset entries.
SCAN_EXCLUDE = {"_synth_test", "_synth_reports", "reports", ".venv", "__pycache__"}

_EXIF_TAG_BY_NAME = {v: k for k, v in ExifTags.TAGS.items()}


@dataclass
class Score:
    brightness_mean: float
    brightness_std: float
    highlight_clipping_ratio: float        # fraction of pixels with luminance > 250
    highlight_retention: float             # 1.0 = matches darkest bracket's top-1% structure
    edge_energy: float                     # Laplacian variance on luminance (proxy, not "sharpness")
    oversaturation_ratio: float
    color_delta_e: Optional[float]


@dataclass
class FusionResult:
    method: str
    path: Path
    score: Score
    seconds: float
    note: str = ""


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTS


def read_image(path: Path) -> np.ndarray:
    """Decode any supported image to an 8-bit BGR ndarray with EXIF orientation applied."""
    if is_raw(path):
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=False,
            )
        # rgb is RGB; convert to BGR for OpenCV consumers.
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Use PIL for orientation handling + ICC awareness, then hand off to numpy/BGR.
    with Image.open(path) as pil:
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def write_image(img: np.ndarray, path: Path, jpeg_quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    else:
        ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cannot encode {path}")
    buf.tofile(str(path))


def read_exposure_time(path: Path) -> Optional[float]:
    """Return ExposureTime in seconds, or None."""
    try:
        if is_raw(path):
            # rawpy doesn't surface EXIF; fall back to PIL on a stripped JPEG preview if needed.
            # For now return None; bracket order then falls back to filename sort.
            return None
        with Image.open(path) as pil:
            exif = pil._getexif() or {}
            tag = _EXIF_TAG_BY_NAME.get("ExposureTime")
            v = exif.get(tag) if tag else None
            if v is None:
                return None
            if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
                return v[0] / v[1]
            if isinstance(v, (int, float)):
                return float(v)
            # Pillow exposes ExifTags.IFDRational; cast via float().
            try:
                return float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
    except Exception:
        return None


def load_brackets(brackets_dir: Path) -> list[Path]:
    if not brackets_dir.is_dir():
        raise ValueError(f"brackets dir not found: {brackets_dir}")
    files = sorted(p for p in brackets_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    # Drop flash-only frames; they belong to the Flambient path, not HDR fusion.
    files = [p for p in files if "_flash" not in p.stem.lower()]
    if len(files) < 2:
        raise ValueError(f"need at least 2 bracket frames in {brackets_dir}, got {len(files)}")
    # Sort by EXIF ExposureTime when available (darkest → brightest); fall back to filename order.
    timed = [(read_exposure_time(p), p) for p in files]
    if all(t is not None for t, _ in timed):
        timed.sort(key=lambda x: x[0] or 0.0)
        files = [p for _, p in timed]
    return files


def fuse_mertens(bracket_paths: list[Path]) -> np.ndarray:
    images = [read_image(p) for p in bracket_paths]
    shapes = {(img.shape[1], img.shape[0]) for img in images}
    if len(shapes) > 1:
        sizes = ", ".join(f"{p.name}={img.shape[1]}x{img.shape[0]}" for p, img in zip(bracket_paths, images))
        raise ValueError(
            f"bracket frames have mismatched dimensions ({sizes}). "
            f"MergeMertens requires identical sizes; brackets must come from the same camera/shoot."
        )
    # MergeMertens documented to accept uint8 BGR; pass as-is.
    merger = cv2.createMergeMertens()
    fused = merger.process(images)
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def fuse_enfuse(bracket_paths: list[Path], tmp_dir: Path) -> tuple[Optional[np.ndarray], str]:
    """Returns (image_or_None, reason_if_none)."""
    if not shutil.which("enfuse"):
        return None, "binary not on PATH"
    out_path = tmp_dir / "enfuse_out.tif"
    # Use `--` to prevent any filename starting with - being parsed as an option.
    cmd = ["enfuse", "-o", str(out_path), "--", *[str(p) for p in bracket_paths]]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "timed out after 300s"
    except OSError as e:
        return None, f"spawn failed: {e}"
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-200:]
        return None, f"exit {proc.returncode}: {tail.strip()}"
    try:
        return read_image(out_path), ""
    except IOError as e:
        return None, f"output unreadable: {e}"


def darkest_bracket(bracket_paths: list[Path]) -> np.ndarray:
    """Return the bracket with the shortest EXIF ExposureTime; fall back to lowest mean luminance."""
    timed: list[tuple[float, Path]] = []
    for p in bracket_paths:
        t = read_exposure_time(p)
        if t is not None:
            timed.append((t, p))
    if len(timed) == len(bracket_paths):
        timed.sort(key=lambda x: x[0])
        return read_image(timed[0][1])
    # Fallback: pick by mean luminance.
    darkest_img, lowest = None, float("inf")
    for p in bracket_paths:
        img = read_image(p)
        m = float(img.mean())
        if m < lowest:
            lowest = m
            darkest_img = img
    assert darkest_img is not None
    return darkest_img


def score_image(fused: np.ndarray, darkest: np.ndarray, reference: Optional[np.ndarray]) -> Score:
    gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV)

    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    # Highlight clipping: fraction of near-white pixels (lost detail in windows/sky).
    highlight_clipping_ratio = float((gray > 250).mean())

    # Highlight retention: structural similarity in the brightest 1% of the darkest bracket.
    # Closer to 1.0 = better preserved window detail.
    darkest_gray = cv2.cvtColor(darkest, cv2.COLOR_BGR2GRAY)
    cutoff = float(np.percentile(darkest_gray, 99))
    mask = darkest_gray >= cutoff
    if mask.sum() > 0 and fused.shape[:2] == darkest.shape[:2]:
        fused_in_mask = gray[mask].astype(np.float32)
        dark_in_mask = darkest_gray[mask].astype(np.float32)
        # Penalise both blown highlights (fused much brighter) and lost detail (fused much darker).
        ratio = fused_in_mask.mean() / max(dark_in_mask.mean(), 1.0)
        highlight_retention = float(max(0.0, 1.0 - abs(np.log(max(ratio, 1e-3)))))
    else:
        highlight_retention = 0.0

    edge_energy = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    oversaturation_ratio = float((hsv[..., 1] > 242).mean())

    color_delta_e: Optional[float] = None
    if reference is not None:
        if fused.shape[:2] != reference.shape[:2]:
            fused_r = cv2.resize(fused, (reference.shape[1], reference.shape[0]))
        else:
            fused_r = fused
        # Downsample for Lab conversion speed on 24MP images.
        max_dim = 1024
        h, w = reference.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            ref_small = cv2.resize(reference, (int(w * scale), int(h * scale)))
            fused_small = cv2.resize(fused_r, (int(w * scale), int(h * scale)))
        else:
            ref_small, fused_small = reference, fused_r
        lab_a = color.rgb2lab(cv2.cvtColor(fused_small, cv2.COLOR_BGR2RGB) / 255.0)
        lab_b = color.rgb2lab(cv2.cvtColor(ref_small, cv2.COLOR_BGR2RGB) / 255.0)
        color_delta_e = float(np.linalg.norm(lab_a - lab_b, axis=2).mean())

    return Score(
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        highlight_clipping_ratio=highlight_clipping_ratio,
        highlight_retention=highlight_retention,
        edge_energy=edge_energy,
        oversaturation_ratio=oversaturation_ratio,
        color_delta_e=color_delta_e,
    )


def find_first_image(dir_path: Path) -> Optional[Path]:
    if not dir_path.is_dir():
        return None
    candidates = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    return candidates[0] if candidates else None


def benchmark_room(
    room_dir: Path,
    out_dir: Path,
    slug: str,
    include_reference: bool,
    include_autohdr: bool,
    run_tmp_root: Path,
) -> list[FusionResult]:
    try:
        rel = room_dir.relative_to(Path.cwd())
        display = str(rel)
    except ValueError:
        display = str(room_dir)
    print(f"\n[{display}]")

    bracket_paths = load_brackets(room_dir / "brackets")
    darkest = darkest_bracket(bracket_paths)
    ref_path = find_first_image(room_dir / "final") if include_reference else None
    reference = read_image(ref_path) if ref_path else None

    results: list[FusionResult] = []
    tmp = run_tmp_root / slug
    tmp.mkdir(parents=True, exist_ok=True)

    # Mertens
    t0 = time.perf_counter()
    try:
        mertens = fuse_mertens(bracket_paths)
        mertens_path = out_dir / f"{slug}_mertens.jpg"
        write_image(mertens, mertens_path)
        results.append(FusionResult(
            method="mertens",
            path=mertens_path,
            score=score_image(mertens, darkest, reference),
            seconds=time.perf_counter() - t0,
        ))
        print(f"  mertens: {results[-1].seconds:.2f}s")
    except (cv2.error, ValueError, MemoryError) as e:
        print(f"  mertens: failed ({e})", file=sys.stderr)

    # Enfuse
    t0 = time.perf_counter()
    enfuse, reason = fuse_enfuse(bracket_paths, tmp)
    if enfuse is not None:
        enfuse_path = out_dir / f"{slug}_enfuse.jpg"
        write_image(enfuse, enfuse_path)
        results.append(FusionResult(
            method="enfuse",
            path=enfuse_path,
            score=score_image(enfuse, darkest, reference),
            seconds=time.perf_counter() - t0,
        ))
        print(f"  enfuse:  {results[-1].seconds:.2f}s")
    else:
        print(f"  enfuse:  skipped ({reason})")

    # AutoHDR comparison (already-produced output)
    if include_autohdr:
        autohdr_path = find_first_image(room_dir / "autohdr_output")
        if autohdr_path:
            try:
                autohdr = read_image(autohdr_path)
                results.append(FusionResult(
                    method="autohdr",
                    path=autohdr_path,
                    score=score_image(autohdr, darkest, reference),
                    seconds=0.0,
                    note="pre-computed by AutoHDR",
                ))
                print("  autohdr: pre-computed")
            except IOError as e:
                print(f"  autohdr: unreadable ({e})", file=sys.stderr)

    # Reference (Alan's Lightroom final). Reuse loaded image.
    if include_reference and ref_path:
        results.append(FusionResult(
            method="reference",
            path=ref_path,
            score=score_image(reference, darkest, None),
            seconds=0.0,
            note="Alan's Lightroom final — ΔE is omitted (compared against itself)",
        ))
        print(f"  reference: {ref_path.name}")

    return results


def render_html(room_dir: Path, results: list[FusionResult], out_path: Path) -> None:
    out_parent = out_path.parent.resolve()
    cards = []
    for r in results:
        rp = r.path.resolve()
        try:
            rel = rp.relative_to(out_parent)
            src = urllib.parse.quote(rel.as_posix())
        except ValueError:
            src = urllib.parse.quote(rp.as_posix())
        score_html = "".join(
            f"<div><span>{html.escape(k)}</span><span>{'' if v is None else html.escape(f'{v:.3f}')}</span></div>"
            for k, v in asdict(r.score).items()
        )
        note_html = f'<div class="note">{html.escape(r.note)}</div>' if r.note else ""
        cards.append(
            f'<div class="card"><h2>{html.escape(r.method)} — {r.seconds:.2f}s</h2>'
            f'<img src="{src}" alt="{html.escape(r.method)}">'
            f'<div class="scores">{score_html}</div>{note_html}</div>'
        )
    title = html.escape(str(room_dir))
    style = (
        "body{font-family:-apple-system,sans-serif;background:#111;color:#eee;padding:20px}"
        "h1{font-size:18px;margin-bottom:16px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}"
        ".card{background:#1a1a1a;border:1px solid #333;padding:12px;border-radius:6px}"
        ".card h2{font-size:14px;margin:0 0 8px;color:#00d4b4}"
        ".card img{width:100%;height:auto;border-radius:4px;display:block}"
        ".scores{font-size:11px;margin-top:8px}"
        ".scores div{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #222}"
        ".scores span:last-child{color:#aaa;font-variant-numeric:tabular-nums}"
        ".note{font-size:10px;color:#888;margin-top:6px;font-style:italic}"
    )
    body = "\n".join(cards)
    html_doc = (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
        f'<style>{style}</style></head><body>'
        f'<h1>{title}</h1><div class="grid">{body}</div></body></html>'
    )
    out_path.write_text(html_doc, encoding="utf-8")


def discover_rooms(target: Path) -> list[Path]:
    """Find every room dir (one containing brackets/) under target. Excludes scratch dirs."""
    if (target / "brackets").is_dir():
        return [target]
    rooms: list[Path] = []
    for p in target.rglob("brackets"):
        if not p.is_dir():
            continue
        # Skip if any path component is in SCAN_EXCLUDE.
        if any(part in SCAN_EXCLUDE or part.startswith(".") for part in p.parts):
            continue
        rooms.append(p.parent)
    return sorted(rooms)


def slug_for(room: Path, base: Path) -> str:
    try:
        parts = room.relative_to(base).parts
    except ValueError:
        parts = (room.name,)
    return "_".join(parts) or room.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path, help="A room dir (contains brackets/) or a dataset root.")
    parser.add_argument("--report-out", type=Path, default=Path("reports"))
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--include-autohdr", action="store_true")
    args = parser.parse_args()

    target: Path = args.target.resolve()
    out: Path = args.report_out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    room_dirs = discover_rooms(target)
    if not room_dirs:
        print(f"no bracket sets found under {target}", file=sys.stderr)
        return 1

    # Single shared temp root for the whole run, auto-cleaned.
    with tempfile.TemporaryDirectory(prefix="benchmark_") as tmp_root_str:
        run_tmp_root = Path(tmp_root_str)

        # When target is a single room, slugs are relative to its parent.
        base = target if target.is_dir() and (target / "brackets").is_dir() is False else target.parent

        all_results: list[tuple[Path, str, list[FusionResult]]] = []
        for room in room_dirs:
            slug = slug_for(room, base)
            try:
                results = benchmark_room(
                    room, out, slug,
                    args.include_reference, args.include_autohdr,
                    run_tmp_root,
                )
                if results:
                    render_html(room, results, out / f"{slug}.html")
                    all_results.append((room, slug, results))
            except (ValueError, IOError, OSError, cv2.error, MemoryError) as e:
                print(f"  skipped: {e}", file=sys.stderr)

    # Write summary CSV.
    summary_path = out / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["room", "slug", "method", "seconds", *Score.__dataclass_fields__.keys()])
        for room, slug, results in all_results:
            for r in results:
                w.writerow([
                    str(room),
                    slug,
                    r.method,
                    f"{r.seconds:.3f}",
                    *[f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)
                      for v in asdict(r.score).values()],
                ])

    print(f"\nReports: {out}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
