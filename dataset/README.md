# PhotoHDR MV — Dataset

Holds bracket sets + finals + masks from Alan's MV shoots. Used for Phase 0b benchmarking, Phase 2 prompt validation, and Phase 3 fine-tuning.

This folder is intentionally empty until brackets arrive on the dev box. The structure below is the contract — populate it the same way every property to keep tooling simple.

## Folder structure

```
dataset/
  property_0001/
    room_001/
      brackets/
        IMG_0001_-2.CR3      ← dark exposure
        IMG_0002_0.CR3       ← base / ambient
        IMG_0003_+2.CR3      ← bright exposure
        IMG_0004_flash.CR3   ← (flambient only) flash frame
      final/
        IMG_0002_final.jpg   ← Alan's finished photo (Lightroom export)
      masks/
        window.png
        sky.png
        grass.png
        tv.png
        fireplace.png
      meta.json
    room_002/
      ...
  property_0002/
    ...
```

### Filename conventions

- **Brackets:** use the original camera filename with an EV suffix: `IMG_NNNN_-2.CR3`, `IMG_NNNN_0.CR3`, `IMG_NNNN_+2.CR3`. EV `0` is the base/ambient.
- **Flambient:** the flash-only frame is `IMG_NNNN_flash.<ext>`. Ambient frame is `IMG_NNNN_0.<ext>`.
- **Final:** keep Alan's finished JPEG named after the base bracket: `IMG_NNNN_final.jpg`.
- **Masks:** PNG with alpha. One file per semantic region. Lower case, no spaces.
- **Property + room IDs:** zero-padded 4-digit (`property_0001`, `room_001`). Increment monotonically across the dataset.

### meta.json

One per room. See [`_template/room/meta.json`](_template/room/meta.json) for the canonical fields. Required:

| Field | Type | Notes |
|---|---|---|
| `property_id` | string | Matches the folder name |
| `room_type` | string | `living_room` / `kitchen` / `bedroom` / `bathroom` / `exterior` / `aerial` / `twilight` |
| `shoot_type` | string | `hdr` \| `flambient` \| `single` |
| `bracket_count` | int | Number of bracket frames (excluding flash) |
| `ev_spacing` | int | EV step between brackets (typically 2) |
| `camera` | string | Body model |
| `lens` | string | Lens model |
| `flash_used` | bool | True for flambient |
| `editor` | string | Who finished it (`alan` for now) |
| `style` | string | Free-form style label, e.g., `natural_real_estate` |
| `notes` | string | Anything else (location, lighting conditions, problems) |

## Phase 0b — benchmark dataset

Target for Phase 0b is **5-10 real bracket sets**. Each must include:
- Brackets in `brackets/`
- Alan's Lightroom-finished version in `final/`

Masks are NOT required for Phase 0b — benchmarking only compares fusion quality against AutoHDR, Lightroom HDR Merge, and Enfuse. Masks become required for Phase 2 (targeted ops) onward.

## Phase target sizes (from v3 spec, Section E3)

| Phase | Pairs needed |
|---|---|
| Phase 0b | 5-10 (benchmark gate) |
| Phase 1 | 0 (off-the-shelf APIs only) |
| Phase 2 | 400-500 (validate generative prompts) |
| Phase 3 | 20,000+ (NAFNet fine-tune) |

## What NOT to put here

- MLS listing photos Alan does not own (VHT v. Zillow ruling — copyright applies regardless of use).
- Scraped images from Zillow, Redfin, or any listing portal.
- Any image without a known license / provenance.

Public datasets that can supplement (separate folder, not under `dataset/`):
- MIT-Adobe FiveK (`data.csail.mit.edu/graphics/fivek/`)
- SICE (`github.com/csjcai/SICE`)
- S2R-HDR (`huggingface.co/datasets/iimmortall/S2R-HDR`)

## Git policy

This folder is committed structure-only. Image files (`*.CR3`, `*.NEF`, `*.ARW`, `*.DNG`, `*.jpg`, `*.png`, `*.tif`) are gitignored — see `.gitignore` in this folder. The dataset lives on local disk + a backup, not in git.

`meta.json` files ARE committed. They are small text and serve as the dataset index.
