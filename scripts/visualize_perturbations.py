from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from robust_vora.data.synthetic_perturbation_dataset import apply_perturbation


PERTURBATIONS = (
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "jpeg",
    "low_light",
    "rain",
    "snow",
    "noise+jpeg",
    "blur+jpeg",
    "low_light+noise",
    "rain+blur",
    "snow+jpeg",
    "rain+low_light+noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize every synthetic perturbation next to the clean crop.")
    parser.add_argument("--image", default="data/Flickr2K/000263.png")
    parser.add_argument("--output-dir", default="outputs/perturbation_preview")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--severity", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = load_center_crop(Path(args.image), args.patch_size)
    rows = []
    labels = []
    for perturbation in PERTURBATIONS:
        degraded = apply_perturbation(
            clean,
            perturbation,
            severity=args.severity,
            rng=random.Random(args.seed),
        ).clamp(0.0, 1.0)
        rows.extend([clean, degraded])
        labels.append(("clean", perturbation))

    raw_grid = output_dir / "all_perturbations_unlabeled.png"
    output_grid = output_dir / "all_perturbations_grid.png"
    save_image(torch.stack(rows, dim=0), raw_grid, nrow=2, padding=8)
    add_row_labels(raw_grid, output_grid, labels)
    raw_grid.unlink(missing_ok=True)

    for perturbation in PERTURBATIONS:
        degraded = apply_perturbation(
            clean,
            perturbation,
            severity=args.severity,
            rng=random.Random(args.seed),
        ).clamp(0.0, 1.0)
        save_image(torch.stack([clean, degraded], dim=0), output_dir / f"{safe_name(perturbation)}.png", nrow=2)

    print(f"saved grid: {output_grid}")
    print(f"saved individual pairs: {output_dir}")


def load_center_crop(path: Path, patch_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image)
    _, height, width = tensor.shape
    crop = min(patch_size, height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    return tensor[:, top : top + crop, left : left + crop]


def add_row_labels(input_path: Path, output_path: Path, labels: list[tuple[str, str]]) -> None:
    image = Image.open(input_path).convert("RGB")
    row_count = len(labels)
    cell_width = image.width // 2
    row_height = image.height // row_count
    label_width = 180
    header_height = 26
    canvas = Image.new("RGB", (image.width + label_width, image.height + header_height), "white")
    canvas.paste(image, (label_width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((label_width + 8, 7), "clean", fill="black")
    draw.text((label_width + cell_width + 8, 7), "perturbed", fill="black")
    for row, (_clean_label, perturbation) in enumerate(labels):
        y = header_height + row * row_height + max(2, math.floor(row_height * 0.42))
        draw.text((8, y), perturbation, fill="black")
    canvas.save(output_path)


def safe_name(name: str) -> str:
    return name.replace("+", "_plus_").replace("/", "_")


if __name__ == "__main__":
    main()
