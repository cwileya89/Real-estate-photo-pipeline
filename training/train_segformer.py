"""SegFormer-b0 window-mask fine-tune.

Trains a binary segmentation model on interior window masks. Fits in 16GB VRAM at
512×512 input with batch 8. Uses HuggingFace transformers + datasets.

Status: structurally complete; cannot run until ≥1,000 annotated window masks exist.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
except ImportError as e:
    print(f"missing dep: {e}. Install requirements first.", file=sys.stderr)
    sys.exit(1)


PRETRAINED_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


@dataclass
class Config:
    images_root: Path
    output_dir: Path
    epochs: int
    batch_size: int = 8
    image_size: int = 512
    learning_rate: float = 5e-5
    num_workers: int = 4


class WindowMaskDataset(Dataset):
    """Each sample folder contains image.jpg + mask.png (white=window, black=not).

    Uses the SegformerImageProcessor for proper input normalisation (matches the
    pretrained model's expected mean/std). Mask resize is NEAREST to avoid
    bilinear blending across boundary pixels.
    """

    def __init__(self, root: Path, image_size: int, processor: SegformerImageProcessor):
        self.image_size = image_size
        self.processor = processor
        self.samples: list[tuple[Path, Path]] = []
        skipped = 0
        for sample_dir in sorted(root.iterdir()):
            if not sample_dir.is_dir():
                continue
            img = sample_dir / "image.jpg"
            mask = sample_dir / "mask.png"
            if img.exists() and mask.exists():
                self.samples.append((img, mask))
            else:
                skipped += 1
        if not self.samples:
            raise FileNotFoundError(f"no image/mask pairs under {root}")
        if skipped:
            print(f"  skipped {skipped} incomplete sample dirs", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Image goes through the processor for the model's expected normalisation.
        inputs = self.processor(images=img, return_tensors="pt", size={"height": self.image_size, "width": self.image_size})
        img_t = inputs["pixel_values"].squeeze(0)
        # Mask must use NEAREST to preserve binary boundaries before thresholding.
        mask_resized = resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        mask_t = torch.from_numpy(np.array(mask_resized)).long()
        mask_t = (mask_t > 127).long()  # binarise
        return img_t, mask_t


def train(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device; SegFormer fine-tune on CPU is impractical", file=sys.stderr)

    processor = SegformerImageProcessor.from_pretrained(PRETRAINED_ID)
    dataset = WindowMaskDataset(cfg.images_root, cfg.image_size, processor)
    print(f"loaded {len(dataset)} image/mask pairs")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_ID,
        num_labels=2,
        id2label={0: "background", 1: "window"},
        label2id={"background": 0, "window": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(pixel_values=imgs, labels=masks)
            optimizer.zero_grad(set_to_none=True)
            outputs.loss.backward()
            optimizer.step()
            running += float(outputs.loss.item()) * imgs.size(0)
            n += imgs.size(0)
        avg = running / max(n, 1)
        print(f"epoch {epoch + 1}/{cfg.epochs} loss={avg:.4f}")

        ckpt_dir = cfg.output_dir / f"epoch_{epoch + 1:03d}"
        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)  # keep preprocessing consistent at inference time
            print(f"  → saved {ckpt_dir}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True, help="root with sample folders (image.jpg + mask.png)")
    p.add_argument("--output", type=Path, default=Path("runs/segformer_b0"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    cfg = Config(
        images_root=args.images.resolve(),
        output_dir=args.output.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
