"""LightFuse exposure-fusion fine-tune.

Trains a learned exposure-fusion network on (brackets → reference fused) tuples.
Reference is typically Alan's Lightroom HDR Merge or AutoHDR output. Patch-based
training at 256×256 fits the RTX 5070 Ti.

Status: structurally complete; cannot run until Phase 0b benchmark establishes that
Enfuse fusion is insufficient AND ≥500 bracket+reference tuples exist.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
except ImportError as e:
    print(f"missing dep: {e}. Install requirements first.", file=sys.stderr)
    sys.exit(1)


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class Config:
    brackets_root: Path
    references_root: Path
    output_dir: Path
    epochs: int
    num_brackets: int = 3
    batch_size: int = 8
    patch_size: int = 256
    learning_rate: float = 1e-4
    num_workers: int = 4


class BracketReferenceDataset(Dataset):
    """Each sample folder under brackets/ contains exactly num_brackets bracket frames;
    references/<id>/fused.jpg holds the target. Mismatched bracket counts are rejected
    upfront — DataLoader collation requires identical tensor shapes."""

    def __init__(self, brackets_root: Path, references_root: Path, patch_size: int, num_brackets: int):
        self.patch_size = patch_size
        self.num_brackets = num_brackets
        self.samples: list[tuple[list[Path], Path]] = []
        skipped = 0
        for room_dir in sorted(brackets_root.rglob("brackets")):
            if not room_dir.is_dir():
                continue
            bracket_files = sorted(p for p in room_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
            if len(bracket_files) != num_brackets:
                skipped += 1
                continue
            ref_dir = references_root / room_dir.parent.relative_to(brackets_root)
            ref = ref_dir / "fused.jpg"
            if not ref.exists():
                skipped += 1
                continue
            self.samples.append((bracket_files, ref))
        if not self.samples:
            raise FileNotFoundError(
                f"no bracket+reference samples found under {brackets_root} "
                f"with exactly {num_brackets} bracket frames"
            )
        if skipped:
            print(f"  skipped {skipped} folders (wrong bracket count or missing reference)", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        bracket_paths, ref_path = self.samples[idx]
        brackets = [Image.open(p).convert("RGB") for p in bracket_paths]
        ref = Image.open(ref_path).convert("RGB")
        ps = self.patch_size
        w, h = ref.size
        if w < ps or h < ps:
            raise ValueError(f"reference too small for {ps}px patch: {ref_path}")
        x = torch.randint(0, w - ps + 1, (1,)).item()
        y = torch.randint(0, h - ps + 1, (1,)).item()
        bracket_tensors = [to_tensor(b.resize(ref.size).crop((x, y, x + ps, y + ps))) for b in brackets]
        ref_t = to_tensor(ref.crop((x, y, x + ps, y + ps)))
        # Stack brackets along channel dim: [N*3, H, W]
        stacked = torch.cat(bracket_tensors, dim=0)
        return stacked, ref_t


class TinyFusionBaseline(nn.Module):
    """Smoke-test baseline: shallow conv stack with no down/up-sampling and no skip
    connections. NOT LightFuse. Wire in the real LightFuse repo (or DeepFuse etc.) once
    available; this baseline is only here to prove the data path runs end-to-end."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.bottleneck(self.encoder(x)))


def train(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device; training will be impractical on CPU", file=sys.stderr)

    dataset = BracketReferenceDataset(cfg.brackets_root, cfg.references_root, cfg.patch_size, cfg.num_brackets)
    print(f"loaded {len(dataset)} bracket+reference samples ({cfg.num_brackets} brackets each)")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )

    in_channels = cfg.num_brackets * 3
    model = TinyFusionBaseline(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.L1Loss()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        avg = running / max(n, 1)
        print(f"epoch {epoch + 1}/{cfg.epochs} loss={avg:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            ckpt = cfg.output_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "loss": avg}, ckpt)
            print(f"  → saved {ckpt}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brackets", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/lightfuse"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num-brackets", type=int, default=3, help="exact bracket count per sample; samples with other counts are skipped")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    cfg = Config(
        brackets_root=args.brackets.resolve(),
        references_root=args.references.resolve(),
        output_dir=args.output.resolve(),
        epochs=args.epochs,
        num_brackets=args.num_brackets,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
