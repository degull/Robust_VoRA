from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from robust_vora.data import SyntheticPerturbationDataset
from robust_vora.models import SimpleRestorationCNN, create_adaptive_vora_restormer, create_restormer
from robust_vora.training import evaluate


PERTURBATIONS = (
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "jpeg",
    "low_light",
    "rain",
    "snow",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a restoration model per synthetic perturbation.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-csv", default="outputs/eval/by_perturbation.csv")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-valid-steps", type=int, default=10)
    parser.add_argument("--model", choices=["simple", "restormer", "adaptive_vora"], default="restormer")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--restormer-root", default="third_party/Restormer")
    parser.add_argument("--restormer-checkpoint", default=None)
    parser.add_argument("--restormer-layernorm", choices=["BiasFree", "WithBias"], default="WithBias")
    parser.add_argument("--vora-rank", type=int, default=8)
    parser.add_argument("--vora-code-dim", type=int, default=64)
    parser.add_argument("--vora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"checkpoint: {args.restormer_checkpoint or args.checkpoint or 'none'}")
    print()
    print(f"{'perturbation':<18} {'loss':>10} {'psnr':>10} {'ssim':>10} {'steps':>8}")
    print("-" * 62)

    rows = []
    for perturbation in PERTURBATIONS:
        dataset = SyntheticPerturbationDataset(
            roots=[Path(args.data_root) / "DIV2K_valid_HR"],
            patch_size=args.patch_size,
            perturbations=[perturbation],
            training=False,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        metrics = evaluate(model, loader, device, max_steps=args.max_valid_steps)
        row = {
            "perturbation": perturbation,
            "loss": metrics["loss"],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "steps": int(metrics["steps"]),
        }
        rows.append(row)
        print(
            f"{perturbation:<18} "
            f"{row['loss']:>10.4f} "
            f"{row['psnr']:>10.2f} "
            f"{row['ssim']:>10.4f} "
            f"{row['steps']:>8d}"
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["perturbation", "loss", "psnr", "ssim", "steps"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"saved: {output_csv}")


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "simple":
        model = SimpleRestorationCNN(channels=args.channels, num_blocks=args.num_blocks)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state)
        return model

    if args.model == "restormer":
        return create_restormer(
            restormer_root=args.restormer_root,
            checkpoint=args.restormer_checkpoint,
            freeze=True,
            layer_norm_type=args.restormer_layernorm,
        )

    model = create_adaptive_vora_restormer(
        restormer_root=args.restormer_root,
        checkpoint=args.restormer_checkpoint,
        layer_norm_type=args.restormer_layernorm,
        rank=args.vora_rank,
        code_dim=args.vora_code_dim,
        adapter_scale=args.vora_scale,
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)
    return model


if __name__ == "__main__":
    main()
