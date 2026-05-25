# Phase 0b Benchmark Harness

Compares HDR fusion approaches on Alan's bracket sets to establish the "acceptable to Alan" quality floor before any pipeline code ships.

**STATUS:** Cannot run yet — Alan's bracket sets are not on the dev box. The harness is ready; drop brackets into `../dataset/property_*/room_*/brackets/` and run.

## What it compares

For each bracket set:

1. **OpenCV MergeMertens** — pure Python, no external binary. The baseline our Phase 1 worker will start with.
2. **Enfuse** — Hugin's exposure-fusion tool. Industry standard for "natural" HDR. Requires `enfuse` on PATH.
3. **Alan's Lightroom HDR Merge** — pre-fused DNG/JPEG dropped in `final/` by Alan. Treated as reference truth.
4. **AutoHDR** — pre-processed via the free tier and dropped in `autohdr_output/` by Alan. The competitive bar.

## What it scores

| Metric | What it measures | Direction |
|---|---|---|
| `brightness_mean` | overall exposure | match ground truth |
| `brightness_std` | uniformity / blown highlights | lower better |
| `window_pull_retention` | brightness of top-1% pixels vs darkest-bracket top-1% | closer to 1.0 = preserved detail |
| `laplacian_variance` | sharpness / blur | higher better |
| `oversaturation_ratio` | fraction of pixels with sat>0.95 | lower better |
| `color_delta_E` | mean ΔE vs Alan's final (CIELab) | lower better (requires Alan's final) |

These are objective proxies. Final go/no-go for Phase 1 is Alan's eyeball on the comparison report — the scores narrow the field, they don't decide.

## Usage

```bash
cd benchmark/
python -m venv .venv
.venv/Scripts/activate     # Windows
pip install -r requirements.txt

# Single bracket set
python benchmark.py ../dataset/property_0001/room_001/

# Whole dataset
python benchmark.py ../dataset/ --report-out reports/

# Comparing against Alan's Lightroom final and AutoHDR output
# (requires final/ and autohdr_output/ subfolders alongside brackets/)
python benchmark.py ../dataset/ --include-reference --include-autohdr
```

Output: `reports/<property>_<room>.html` with side-by-side comparison + scores, and `reports/summary.csv` aggregated across the run.

## Requirements

- Python 3.10+
- `enfuse` on PATH for the Enfuse method (install Hugin from `hugin-2024.0.1`).
  On Windows: install Hugin, then add `C:\Program Files\Hugin\bin` to PATH.
  On Mac: `brew install hugin`.
  Without enfuse, that method is skipped (not a hard failure).
- Alan's bracket sets in the `../dataset/` structure documented at `../dataset/README.md`.

## What this does NOT do

- It does not fuse for production. Phase 1 worker (`../photo-editor/worker/`) does that.
- It does not auto-detect "best" method. The HTML report shows Alan side-by-side comparisons; he picks.
- It does not measure subjective quality (Alan's eye does that).
