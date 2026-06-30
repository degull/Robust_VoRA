from pathlib import Path

import torch
from torchvision.utils import save_image

from robust_vora.data import build_synthetic_perturbation_loaders


def main() -> None:
    train_loader, valid_loader = build_synthetic_perturbation_loaders(
        data_root="data",
        patch_size=128,
        batch_size=8,
        num_workers=0,
        seed=42,
    )

    print(f"train images: {len(train_loader.dataset)}")
    print(f"valid images: {len(valid_loader.dataset)}")

    batch = next(iter(train_loader))
    print(f"clean: {tuple(batch['clean'].shape)}")
    print(f"degraded: {tuple(batch['degraded'].shape)}")
    print(f"perturbations: {list(batch['perturbation'])}")
    print(f"severity: {batch['severity'].tolist()}")

    output_dir = Path("outputs/debug_dataloader")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = torch.stack([batch["degraded"], batch["clean"]], dim=1).flatten(0, 1)
    save_image(comparison, output_dir / "train_degraded_clean_grid.png", nrow=2)
    print(f"saved: {output_dir / 'train_degraded_clean_grid.png'}")


if __name__ == "__main__":
    main()
