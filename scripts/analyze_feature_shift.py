from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from robust_vora.data import SyntheticPerturbationDataset, list_images
from robust_vora.metrics import batch_psnr, batch_ssim
from robust_vora.models import create_adaptive_vora_restormer, create_restormer


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

FEATURE_LAYERS = (
    "model.encoder_level1",
    "model.encoder_level2",
    "model.encoder_level3",
    "model.latent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature shift compensation from Adaptive VoRA.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/feature_shift")
    parser.add_argument("--restormer-root", default="third_party/Restormer")
    parser.add_argument("--restormer-checkpoint", default="checkpoints/restormer/deraining.pth")
    parser.add_argument("--vora-checkpoint", default="outputs/baseline/best.pt")
    parser.add_argument("--restormer-layernorm", choices=["BiasFree", "WithBias"], default="WithBias")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-valid-steps", type=int, default=None)
    parser.add_argument("--eval-severity", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vora-rank", type=int, default=8)
    parser.add_argument("--vora-code-dim", type=int, default=64)
    parser.add_argument("--vora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-samples", type=int, default=1)
    parser.add_argument("--visual-image", default=None)
    parser.add_argument("--visual-auto-select", action="store_true")
    parser.add_argument("--visual-candidates", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_model = create_restormer(
        restormer_root=args.restormer_root,
        checkpoint=args.restormer_checkpoint,
        freeze=True,
        layer_norm_type=args.restormer_layernorm,
    ).to(device)
    vora_model = create_adaptive_vora_restormer(
        restormer_root=args.restormer_root,
        checkpoint=args.restormer_checkpoint,
        layer_norm_type=args.restormer_layernorm,
        rank=args.vora_rank,
        code_dim=args.vora_code_dim,
        adapter_scale=args.vora_scale,
    ).to(device)
    load_model_checkpoint(vora_model, args.vora_checkpoint)
    frozen_model.eval()
    vora_model.eval()

    print(f"device: {device}")
    print(f"frozen checkpoint: {args.restormer_checkpoint}")
    print(f"vora checkpoint: {args.vora_checkpoint}")
    print()
    print(
        f"{'perturbation':<18} {'d_before':>10} {'d_after':>10} "
        f"{'ratio':>8} {'f_psnr':>8} {'v_psnr':>8} {'f_ssim':>8} {'v_ssim':>8}"
    )
    print("-" * 88)

    rows = []
    sample_batches = []
    visual_clean = None
    if args.visual_image:
        visual_clean = load_visual_clean_patch(Path(args.visual_image), args.patch_size)
    elif args.visual_auto_select:
        visual_clean = select_visual_clean_patch(args)
    for perturbation in PERTURBATIONS:
        loader = build_loader(args, perturbation)
        stats = defaultdict(float)
        steps = 0
        for batch in loader:
            clean = batch["clean"].to(device, non_blocking=True)
            degraded = batch["degraded"].to(device, non_blocking=True)

            with torch.no_grad():
                clean_features, clean_restored = extract_features(frozen_model, clean, FEATURE_LAYERS)
                frozen_features, frozen_restored = extract_features(frozen_model, degraded, FEATURE_LAYERS)
                vora_features, vora_restored = extract_features(vora_model, degraded, FEATURE_LAYERS)

                d_before = feature_distance(frozen_features, clean_features)
                d_after = feature_distance(vora_features, clean_features)
                stats["d_before"] += d_before.item()
                stats["d_after"] += d_after.item()
                stats["frozen_psnr"] += batch_psnr(frozen_restored, clean).mean().item()
                stats["vora_psnr"] += batch_psnr(vora_restored, clean).mean().item()
                stats["frozen_ssim"] += batch_ssim(frozen_restored, clean).mean().item()
                stats["vora_ssim"] += batch_ssim(vora_restored, clean).mean().item()
                steps += 1

            if args.max_valid_steps is not None and steps >= args.max_valid_steps:
                break

        row = average_row(perturbation, stats, steps)
        rows.append(row)
        print(
            f"{row['perturbation']:<18} "
            f"{row['d_before']:>10.4f} {row['d_after']:>10.4f} {row['ratio']:>8.3f} "
            f"{row['frozen_psnr']:>8.2f} {row['vora_psnr']:>8.2f} "
            f"{row['frozen_ssim']:>8.4f} {row['vora_ssim']:>8.4f}"
        )

        if visual_clean is not None:
            sample_batches.append(
                build_visual_sample(
                    args,
                    perturbation,
                    visual_clean,
                    frozen_model,
                    vora_model,
                    device,
                )
            )

    save_csv(rows, output_dir / "feature_shift_metrics.csv")
    save_bar_plot(rows, output_dir / "feature_shift_ratio.png")
    save_table_plot(rows, output_dir / "feature_shift_table.png")
    save_sample_grid(sample_batches, output_dir / "restoration_comparison_grid.png")
    print()
    print(f"saved metrics: {output_dir / 'feature_shift_metrics.csv'}")
    print(f"saved plot: {output_dir / 'feature_shift_ratio.png'}")
    print(f"saved table: {output_dir / 'feature_shift_table.png'}")
    print(f"saved images: {output_dir / 'restoration_comparison_grid.png'}")


def build_loader(args: argparse.Namespace, perturbation: str) -> DataLoader:
    dataset = SyntheticPerturbationDataset(
        roots=[Path(args.data_root) / "DIV2K_valid_HR"],
        patch_size=args.patch_size,
        perturbations=[perturbation],
        fixed_severity=args.eval_severity,
        training=False,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def select_visual_clean_patch(args: argparse.Namespace) -> torch.Tensor:
    paths = list_images([Path(args.data_root) / "DIV2K_valid_HR"])
    best_score = -1.0
    best_patch = None
    for path in paths[: args.visual_candidates]:
        image = Image.open(path).convert("RGB")
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3)
        tensor = tensor.permute(2, 0, 1) / 255.0
        patch = center_crop(tensor, args.patch_size)
        score = edge_score(patch)
        if score > best_score:
            best_score = score
            best_patch = patch
    if best_patch is None:
        raise RuntimeError("Failed to select visual patch.")
    print(f"selected visualization patch edge score: {best_score:.4f}")
    return best_patch.unsqueeze(0)


def load_visual_clean_patch(path: Path, patch_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3)
    tensor = tensor.permute(2, 0, 1) / 255.0
    patch = center_crop(tensor, patch_size)
    print(f"selected visualization image: {path}")
    return patch.unsqueeze(0)


def build_visual_sample(args, perturbation: str, clean: torch.Tensor, frozen_model, vora_model, device):
    import random

    from robust_vora.data.synthetic_perturbation_dataset import apply_perturbation

    clean_device = clean.to(device)
    degraded = apply_perturbation(clean_device[0], perturbation, severity=0.9, rng=random.Random(args.seed)).unsqueeze(0)
    with torch.no_grad():
        frozen_restored = frozen_model(degraded)
        vora_restored = vora_model(degraded)
    return {
        "perturbation": perturbation,
        "degraded": degraded.detach().cpu(),
        "frozen": frozen_restored.detach().cpu(),
        "vora": vora_restored.detach().cpu(),
        "clean": clean.detach().cpu(),
    }


def center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = image.shape
    crop = min(size, height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    return image[:, top : top + crop, left : left + crop]


def edge_score(image: torch.Tensor) -> float:
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square()).mean().item()


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Adaptive VoRA checkpoint missing keys: {len(missing)}")
    if unexpected:
        print(f"Adaptive VoRA checkpoint unexpected keys: {len(unexpected)}")


def extract_features(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_names: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())

    for layer_name in layer_names:
        module = modules[layer_name]

        def hook(_module, _inputs, output, name=layer_name):
            features[name] = output.detach()

        handles.append(module.register_forward_hook(hook))

    output = model(image)
    for handle in handles:
        handle.remove()
    return features, output


def feature_distance(
    source_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    distances = []
    for name, source in source_features.items():
        target = target_features[name]
        source_norm = F.normalize(source.flatten(2), dim=1)
        target_norm = F.normalize(target.flatten(2), dim=1)
        distances.append((source_norm - target_norm).square().mean(dim=(1, 2)).sqrt())
    return torch.stack(distances, dim=0).mean()


def average_row(perturbation: str, stats: dict[str, float], steps: int) -> dict[str, float | str]:
    divisor = max(steps, 1)
    d_before = stats["d_before"] / divisor
    d_after = stats["d_after"] / divisor
    return {
        "perturbation": perturbation,
        "d_before": d_before,
        "d_after": d_after,
        "ratio": d_after / max(d_before, 1e-8),
        "frozen_psnr": stats["frozen_psnr"] / divisor,
        "vora_psnr": stats["vora_psnr"] / divisor,
        "frozen_ssim": stats["frozen_ssim"] / divisor,
        "vora_ssim": stats["vora_ssim"] / divisor,
        "steps": steps,
    }


def save_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_bar_plot(rows: list[dict[str, float | str]], output_path: Path) -> None:
    names = [str(row["perturbation"]) for row in rows]
    ratios = [float(row["ratio"]) for row in rows]
    colors = ["#2f7d59" if ratio < 1.0 else "#b24a3b" for ratio in ratios]
    plt.figure(figsize=(10, 4.5))
    plt.bar(names, ratios, color=colors)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.ylabel("Feature Alignment Ratio")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_table_plot(rows: list[dict[str, float | str]], output_path: Path) -> None:
    columns = ["Perturbation", "D_before", "D_after", "Ratio", "Frozen PSNR", "VoRA PSNR"]
    values = [
        [
            str(row["perturbation"]),
            f"{float(row['d_before']):.4f}",
            f"{float(row['d_after']):.4f}",
            f"{float(row['ratio']):.3f}",
            f"{float(row['frozen_psnr']):.2f}",
            f"{float(row['vora_psnr']):.2f}",
        ]
        for row in rows
    ]
    plt.figure(figsize=(11, 3.4))
    plt.axis("off")
    table = plt.table(cellText=values, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8ecef")
            cell.set_text_props(weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_sample_grid(sample_batches: list[dict[str, torch.Tensor | str]], output_path: Path) -> None:
    if not sample_batches:
        return
    rows = []
    for batch in sample_batches:
        rows.extend(
            [
                batch["degraded"][0],
                batch["frozen"][0],
                batch["vora"][0],
                batch["clean"][0],
            ]
        )
    temporary_path = output_path.with_name(output_path.stem + "_unlabeled.png")
    save_image(torch.stack(rows, dim=0).clamp(0.0, 1.0), temporary_path, nrow=4, padding=8)
    add_grid_labels(temporary_path, output_path, sample_batches)
    temporary_path.unlink(missing_ok=True)


def add_grid_labels(input_path: Path, output_path: Path, sample_batches: list[dict[str, torch.Tensor | str]]) -> None:
    image = Image.open(input_path).convert("RGB")
    label_height = 24
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    columns = ["degraded", "frozen", "adaptive vora", "clean"]
    cell_width = image.width // 4
    for column, label in enumerate(columns):
        draw.text((column * cell_width + 8, 6), label, fill="black")
    row_height = image.height // max(len(sample_batches), 1)
    for row, batch in enumerate(sample_batches):
        draw.text((8, label_height + row * row_height + 8), str(batch["perturbation"]), fill="black")
    canvas.save(output_path)


if __name__ == "__main__":
    main()
