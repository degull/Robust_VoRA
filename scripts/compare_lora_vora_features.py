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
from robust_vora.data.synthetic_perturbation_dataset import apply_perturbation
from robust_vora.metrics import batch_psnr, batch_ssim
from robust_vora.models import create_adaptive_lora_restormer, create_adaptive_vora_restormer, create_restormer
from robust_vora.training.feature_alignment import extract_features


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
    parser = argparse.ArgumentParser(description="Compare LoRA and VoRA feature shift compensation.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/compare_lora_vora")
    parser.add_argument("--restormer-root", default="third_party/Restormer")
    parser.add_argument("--restormer-checkpoint", default="checkpoints/restormer/deraining.pth")
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--vora-checkpoint", required=True)
    parser.add_argument("--restormer-layernorm", choices=["BiasFree", "WithBias"], default="WithBias")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-valid-steps", type=int, default=None)
    parser.add_argument("--eval-severity", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-perturbation", default="snow", choices=PERTURBATIONS)
    parser.add_argument("--visual-layer", default="model.encoder_level2", choices=FEATURE_LAYERS)
    parser.add_argument("--visual-image", default=None)
    parser.add_argument("--visual-auto-select", action="store_true")
    parser.add_argument("--visual-candidates", type=int, default=40)
    parser.add_argument("--visual-severity", type=float, default=0.9)
    parser.add_argument("--feature-upsample", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = create_restormer(
        restormer_root=args.restormer_root,
        checkpoint=args.restormer_checkpoint,
        freeze=True,
        layer_norm_type=args.restormer_layernorm,
    ).to(device)
    lora = create_adaptive_lora_restormer(
        restormer_root=args.restormer_root,
        checkpoint=args.restormer_checkpoint,
        layer_norm_type=args.restormer_layernorm,
        rank=args.rank,
        code_dim=args.code_dim,
        adapter_scale=args.adapter_scale,
    ).to(device)
    vora = create_adaptive_vora_restormer(
        restormer_root=args.restormer_root,
        checkpoint=args.restormer_checkpoint,
        layer_norm_type=args.restormer_layernorm,
        rank=args.rank,
        code_dim=args.code_dim,
        adapter_scale=args.adapter_scale,
    ).to(device)
    load_checkpoint(lora, args.lora_checkpoint)
    load_checkpoint(vora, args.vora_checkpoint)
    frozen.eval()
    lora.eval()
    vora.eval()

    rows = evaluate_all(args, frozen, lora, vora, device)
    save_csv(rows, output_dir / "lora_vora_feature_metrics.csv")
    save_ratio_plot(rows, output_dir / "lora_vora_feature_ratio.png")
    save_table_plot(rows, output_dir / "lora_vora_feature_table.png")
    save_visualization(args, frozen, lora, vora, device, output_dir)

    print(f"saved metrics: {output_dir / 'lora_vora_feature_metrics.csv'}")
    print(f"saved ratio plot: {output_dir / 'lora_vora_feature_ratio.png'}")
    print(f"saved table: {output_dir / 'lora_vora_feature_table.png'}")
    print(f"saved feature map: {output_dir / 'feature_map_comparison.png'}")
    print(f"saved restoration grid: {output_dir / 'restoration_comparison.png'}")


def evaluate_all(args, frozen, lora, vora, device) -> list[dict[str, float | str]]:
    print(f"{'perturbation':<18} {'lora_ratio':>10} {'vora_ratio':>10} {'l_psnr':>8} {'v_psnr':>8}")
    print("-" * 62)
    rows = []
    for perturbation in PERTURBATIONS:
        loader = build_loader(args, perturbation)
        stats = defaultdict(float)
        steps = 0
        for batch in loader:
            clean = batch["clean"].to(device, non_blocking=True)
            degraded = batch["degraded"].to(device, non_blocking=True)
            with torch.no_grad():
                clean_features, _ = extract_features(frozen, clean, FEATURE_LAYERS, detach=True)
                frozen_features, frozen_out = extract_features(frozen, degraded, FEATURE_LAYERS, detach=True)
                lora_features, lora_out = extract_features(lora, degraded, FEATURE_LAYERS, detach=True)
                vora_features, vora_out = extract_features(vora, degraded, FEATURE_LAYERS, detach=True)

                d_before = feature_distance(frozen_features, clean_features)
                d_lora = feature_distance(lora_features, clean_features)
                d_vora = feature_distance(vora_features, clean_features)
                stats["d_before"] += d_before.item()
                stats["d_lora"] += d_lora.item()
                stats["d_vora"] += d_vora.item()
                stats["frozen_psnr"] += batch_psnr(frozen_out, clean).mean().item()
                stats["lora_psnr"] += batch_psnr(lora_out, clean).mean().item()
                stats["vora_psnr"] += batch_psnr(vora_out, clean).mean().item()
                stats["frozen_ssim"] += batch_ssim(frozen_out, clean).mean().item()
                stats["lora_ssim"] += batch_ssim(lora_out, clean).mean().item()
                stats["vora_ssim"] += batch_ssim(vora_out, clean).mean().item()
                steps += 1
            if args.max_valid_steps is not None and steps >= args.max_valid_steps:
                break

        row = average_row(perturbation, stats, steps)
        rows.append(row)
        print(
            f"{perturbation:<18} {row['lora_ratio']:>10.3f} {row['vora_ratio']:>10.3f} "
            f"{row['lora_psnr']:>8.2f} {row['vora_psnr']:>8.2f}"
        )
    return rows


def save_visualization(args, frozen, lora, vora, device, output_dir: Path) -> None:
    if args.visual_image:
        clean = load_visual_clean_patch(Path(args.visual_image), args.patch_size).to(device)
        import random

        degraded = apply_perturbation(
            clean[0],
            args.visual_perturbation,
            severity=args.visual_severity,
            rng=random.Random(args.seed),
        ).unsqueeze(0)
    elif args.visual_auto_select:
        clean = select_visual_clean_patch(args).to(device)
        import random

        degraded = apply_perturbation(
            clean[0],
            args.visual_perturbation,
            severity=args.visual_severity,
            rng=random.Random(args.seed),
        ).unsqueeze(0)
    else:
        loader = build_loader(args, args.visual_perturbation)
        batch = next(iter(loader))
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
    with torch.no_grad():
        clean_features, _ = extract_features(frozen, clean, (args.visual_layer,), detach=True)
        frozen_features, frozen_out = extract_features(frozen, degraded, (args.visual_layer,), detach=True)
        lora_features, lora_out = extract_features(lora, degraded, (args.visual_layer,), detach=True)
        vora_features, vora_out = extract_features(vora, degraded, (args.visual_layer,), detach=True)

    clean_map = feature_heatmap(clean_features[args.visual_layer], args.feature_upsample)
    degraded_map = feature_heatmap(frozen_features[args.visual_layer], args.feature_upsample)
    lora_map = feature_heatmap(lora_features[args.visual_layer], args.feature_upsample)
    vora_map = feature_heatmap(vora_features[args.visual_layer], args.feature_upsample)
    diff_degraded = feature_diff_heatmap(frozen_features[args.visual_layer], clean_features[args.visual_layer], args.feature_upsample)
    diff_lora = feature_diff_heatmap(lora_features[args.visual_layer], clean_features[args.visual_layer], args.feature_upsample)
    diff_vora = feature_diff_heatmap(vora_features[args.visual_layer], clean_features[args.visual_layer], args.feature_upsample)

    feature_images = torch.stack([clean_map, degraded_map, lora_map, vora_map], dim=0)
    feature_path = output_dir / "feature_map_comparison_unlabeled.png"
    save_image(feature_images, feature_path, nrow=4, padding=12)
    add_labels(
        feature_path,
        output_dir / "feature_map_comparison.png",
        ["clean F(x)", "degraded F(y)", "LoRA F(y)", "VoRA F(y)"],
    )
    feature_path.unlink(missing_ok=True)

    diff_images = torch.stack([diff_degraded, diff_lora, diff_vora], dim=0)
    diff_path = output_dir / "feature_diff_comparison_unlabeled.png"
    save_image(diff_images, diff_path, nrow=3, padding=12)
    add_labels(
        diff_path,
        output_dir / "feature_diff_comparison.png",
        ["|F(y)-F(x)|", "|LoRA-F(x)|", "|VoRA-F(x)|"],
    )
    diff_path.unlink(missing_ok=True)

    restoration_images = torch.cat(
        [
            degraded.detach().cpu(),
            frozen_out.detach().cpu(),
            lora_out.detach().cpu(),
            vora_out.detach().cpu(),
            clean.detach().cpu(),
        ],
        dim=0,
    )
    restoration_path = output_dir / "restoration_comparison_unlabeled.png"
    save_image(restoration_images.clamp(0.0, 1.0), restoration_path, nrow=5, padding=8)
    add_labels(restoration_path, output_dir / "restoration_comparison.png", ["degraded", "frozen", "LoRA", "VoRA", "clean"])
    restoration_path.unlink(missing_ok=True)


def build_loader(args, perturbation: str) -> DataLoader:
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"{checkpoint_path} missing keys: {len(missing)}")
    if unexpected:
        print(f"{checkpoint_path} unexpected keys: {len(unexpected)}")


def feature_distance(source_features: dict[str, torch.Tensor], target_features: dict[str, torch.Tensor]) -> torch.Tensor:
    distances = []
    for name, source in source_features.items():
        target = target_features[name]
        source_norm = F.normalize(source.flatten(2), dim=1)
        target_norm = F.normalize(target.flatten(2), dim=1)
        distances.append((source_norm - target_norm).square().mean(dim=(1, 2)).sqrt())
    return torch.stack(distances, dim=0).mean()


def feature_heatmap(feature: torch.Tensor, upsample: int = 1) -> torch.Tensor:
    heatmap = feature[0].abs().mean(dim=0, keepdim=True).detach().cpu()
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max().clamp_min(1e-8)
    heatmap = colorize_heatmap(heatmap)
    return upsample_image(heatmap, upsample)


def feature_diff_heatmap(source: torch.Tensor, target: torch.Tensor, upsample: int = 1) -> torch.Tensor:
    source_map = F.normalize(source.flatten(2), dim=1).view_as(source)
    target_map = F.normalize(target.flatten(2), dim=1).view_as(target)
    diff = (source_map - target_map).abs()[0].mean(dim=0, keepdim=True).detach().cpu()
    diff = diff - diff.min()
    diff = diff / diff.max().clamp_min(1e-8)
    heatmap = colorize_heatmap(diff)
    return upsample_image(heatmap, upsample)


def colorize_heatmap(gray: torch.Tensor) -> torch.Tensor:
    gray = gray.clamp(0.0, 1.0)
    red = torch.clamp(1.5 * gray, 0.0, 1.0)
    green = torch.clamp(1.5 - (gray - 0.35).abs() * 3.0, 0.0, 1.0)
    blue = torch.clamp(1.5 * (1.0 - gray), 0.0, 1.0)
    return torch.cat([red, green, blue], dim=0)


def upsample_image(image: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return image
    return F.interpolate(image.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)


def average_row(perturbation: str, stats: dict[str, float], steps: int) -> dict[str, float | str]:
    divisor = max(steps, 1)
    d_before = stats["d_before"] / divisor
    d_lora = stats["d_lora"] / divisor
    d_vora = stats["d_vora"] / divisor
    return {
        "perturbation": perturbation,
        "d_before": d_before,
        "d_lora": d_lora,
        "d_vora": d_vora,
        "lora_ratio": d_lora / max(d_before, 1e-8),
        "vora_ratio": d_vora / max(d_before, 1e-8),
        "frozen_psnr": stats["frozen_psnr"] / divisor,
        "lora_psnr": stats["lora_psnr"] / divisor,
        "vora_psnr": stats["vora_psnr"] / divisor,
        "frozen_ssim": stats["frozen_ssim"] / divisor,
        "lora_ssim": stats["lora_ssim"] / divisor,
        "vora_ssim": stats["vora_ssim"] / divisor,
        "steps": steps,
    }


def save_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_ratio_plot(rows: list[dict[str, float | str]], output_path: Path) -> None:
    names = [str(row["perturbation"]) for row in rows]
    x = torch.arange(len(names)).numpy()
    width = 0.36
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, [float(row["lora_ratio"]) for row in rows], width, label="LoRA", color="#5379a6")
    plt.bar(x + width / 2, [float(row["vora_ratio"]) for row in rows], width, label="VoRA", color="#2f7d59")
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.ylabel("Feature Alignment Ratio")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_table_plot(rows: list[dict[str, float | str]], output_path: Path) -> None:
    columns = ["Perturbation", "LoRA Ratio", "VoRA Ratio", "LoRA PSNR", "VoRA PSNR"]
    values = [
        [
            str(row["perturbation"]),
            f"{float(row['lora_ratio']):.3f}",
            f"{float(row['vora_ratio']):.3f}",
            f"{float(row['lora_psnr']):.2f}",
            f"{float(row['vora_psnr']):.2f}",
        ]
        for row in rows
    ]
    plt.figure(figsize=(10, 3.2))
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


def add_labels(input_path: Path, output_path: Path, labels: list[str]) -> None:
    image = Image.open(input_path).convert("RGB")
    label_height = 24
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    cell_width = image.width // len(labels)
    for column, label in enumerate(labels):
        draw.text((column * cell_width + 8, 6), label, fill="black")
    canvas.save(output_path)


if __name__ == "__main__":
    main()
