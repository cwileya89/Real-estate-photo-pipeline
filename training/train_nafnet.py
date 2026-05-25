"""NAFNet final-polish fine-tune.

Expects paired before/after images at --data root. Each pair folder must contain
`before.jpg` (Phase 1 fusion output) + `after.jpg` (Phase 2 / Lightroom polish).
Patches are cropped at 256×256 during training to fit the RTX 5070 Ti's 16GB VRAM.

This script will NOT run without:
  - PyTorch with CUDA 12.x installed (see requirements.txt)
  - The `nafnet` package (install separately: see README — likely `pip install git+...`)
  - At least 1,000 training pairs (target: 20,000)

Status: structurally complete; cannot run until Phase 1+2 has produced training pairs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms.functional as TF
    from PIL import Image
except ImportError as e:
    print(f"missing dep: {e}. Install requirements first.", file=sys.stderr)
    sys.exit(1)


@dataclass
class Config:
    data_root: Path
    output_dir: Path
    epochs: int
    batch_size: int = 16
    patch_size: int = 256
    learning_rate: float = 1e-4
    num_workers: int = 4
    save_every_epochs: int = 5


class BeforeAfterDataset(Dataset):
    """Pair folders with before.jpg + after.jpg."""

    def __init__(self, root: Path, patch_size: int):
        self.patch_size = patch_size
        self.pairs: list[tuple[Path, Path]] = []
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            before = pair_dir / "before.jpg"
            after = pair_dir / "after.jpg"
            if before.exists() and after.exists():
                self.pairs.append((before, after))
        if not self.pairs:
            raise FileNotFoundError(f"no before/after pairs under {root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        before_path, after_path = self.pairs[idx]
        before = Image.open(before_path).convert("RGB")
        after = Image.open(after_path).convert("RGB")
        if before.size != after.size:
            raise ValueError(f"size mismatch in {before_path.parent}: {before.size} vs {after.size}")
        # Random crop both with the same coords.
        w, h = before.size
        ps = self.patch_size
        if w < ps or h < ps:
            raise ValueError(f"image too small for {ps}px patch: {before_path}")
        x = torch.randint(0, w - ps + 1, (1,)).item()
        y = torch.randint(0, h - ps + 1, (1,)).item()
        before_t = TF.to_tensor(before.crop((x, y, x + ps, y + ps)))
        after_t = TF.to_tensor(after.crop((x, y, x + ps, y + ps)))
        return before_t, after_t


def build_nafnet() -> nn.Module:
    """Construct or load NAFNet. Placeholder — wire to the real package once installed.

    Recommended: `pip install git+https://github.com/megvii-research/NAFNet`
    Then: `from basicsr.models.archs.nafnet_arch import NAFNet`
          `return NAFNet(img_channel=3, width=32, enc_blk_nums=[2,2,4,8], middle_blk_num=12, dec_blk_nums=[2,2,2,2])`
    """
    raise NotImplementedError(
        "NAFNet model class not wired yet — install the basicsr/NAFNet package and update build_nafnet()."
    )


def train(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device detected; training will be impractical on CPU", file=sys.stderr)

    dataset = BeforeAfterDataset(cfg.data_root, cfg.patch_size)
    print(f"loaded {len(dataset)} before/after pairs")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )

    model = build_nafnet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.L1Loss()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for before, after in loader:
            before, after = before.to(device, non_blocking=True), after.to(device, non_blocking=True)
            pred = model(before)
            loss = loss_fn(pred, after)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * before.size(0)
            n += before.size(0)
        scheduler.step()
        avg = running / max(n, 1)
        print(f"epoch {epoch + 1}/{cfg.epochs} loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % cfg.save_every_epochs == 0 or epoch == cfg.epochs - 1:
            ckpt = cfg.output_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "loss": avg}, ckpt)
            print(f"  → saved {ckpt}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="root with pair_<id>/before.jpg + after.jpg")
    p.add_argument("--output", type=Path, default=Path("runs/nafnet"), help="checkpoint dir")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    cfg = Config(
        data_root=args.data.resolve(),
        output_dir=args.output.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
