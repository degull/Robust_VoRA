from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from robust_vora.data import build_synthetic_perturbation_loaders
from robust_vora.models import (
    SimpleRestorationCNN,
    create_adaptive_lora_restormer,
    create_adaptive_vora_restormer,
    create_restormer,
)
from robust_vora.training import evaluate, train_one_epoch, train_one_epoch_with_feature_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small baseline restorer on synthetic perturbations.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-valid-steps", type=int, default=None)
    parser.add_argument("--valid-severity", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--model", choices=["simple", "restormer", "adaptive_lora", "adaptive_vora"], default="simple")
    parser.add_argument("--restormer-root", default="third_party/Restormer")
    parser.add_argument("--restormer-checkpoint", default=None)
    parser.add_argument("--restormer-layernorm", choices=["BiasFree", "WithBias"], default="WithBias")
    parser.add_argument("--unfreeze-restormer", action="store_true")
    parser.add_argument("--vora-rank", type=int, default=8)
    parser.add_argument("--vora-code-dim", type=int, default=64)
    parser.add_argument("--vora-scale", type=float, default=1.0)
    parser.add_argument("--lambda-feat", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = build_synthetic_perturbation_loaders(
        data_root=args.data_root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_fixed_severity=args.valid_severity,
        seed=args.seed,
    )
    if args.model == "simple":
        model = SimpleRestorationCNN(channels=args.channels, num_blocks=args.num_blocks).to(device)
    elif args.model == "restormer":
        if args.restormer_checkpoint is None:
            print("warning: Restormer is running without a pretrained checkpoint.")
        model = create_restormer(
            restormer_root=args.restormer_root,
            checkpoint=args.restormer_checkpoint,
            freeze=not args.unfreeze_restormer,
            layer_norm_type=args.restormer_layernorm,
        ).to(device)
    elif args.model == "adaptive_vora":
        if args.restormer_checkpoint is None:
            print("warning: Adaptive VoRA Restormer is running without a pretrained checkpoint.")
        model = create_adaptive_vora_restormer(
            restormer_root=args.restormer_root,
            checkpoint=args.restormer_checkpoint,
            layer_norm_type=args.restormer_layernorm,
            rank=args.vora_rank,
            code_dim=args.vora_code_dim,
            adapter_scale=args.vora_scale,
        ).to(device)
    else:
        if args.restormer_checkpoint is None:
            print("warning: Adaptive LoRA Restormer is running without a pretrained checkpoint.")
        model = create_adaptive_lora_restormer(
            restormer_root=args.restormer_root,
            checkpoint=args.restormer_checkpoint,
            layer_norm_type=args.restormer_layernorm,
            rank=args.vora_rank,
            code_dim=args.vora_code_dim,
            adapter_scale=args.vora_scale,
        ).to(device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr) if trainable_params else None
    clean_feature_model = None
    if args.lambda_feat > 0.0:
        if args.model not in {"adaptive_lora", "adaptive_vora"}:
            raise ValueError("--lambda-feat is currently supported for adaptive adapter models.")
        clean_feature_model = create_restormer(
            restormer_root=args.restormer_root,
            checkpoint=args.restormer_checkpoint,
            freeze=True,
            layer_norm_type=args.restormer_layernorm,
        ).to(device)

    print(f"device: {device}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"valid samples: {len(valid_loader.dataset)}")
    print(f"trainable params: {sum(parameter.numel() for parameter in trainable_params):,}")
    print(f"lambda_feat: {args.lambda_feat}")

    best_psnr = float("-inf")
    log_path = output_dir / "training_log.csv"
    for epoch in range(1, args.epochs + 1):
        if optimizer is None:
            train_metrics = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "steps": 0.0}
        elif clean_feature_model is not None:
            train_metrics = train_one_epoch_with_feature_alignment(
                model,
                clean_feature_model,
                train_loader,
                optimizer,
                device,
                lambda_feat=args.lambda_feat,
                max_steps=args.max_train_steps,
                log_interval=args.log_interval,
                epoch=epoch,
            )
        else:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_steps=args.max_train_steps,
                log_interval=args.log_interval,
                epoch=epoch,
            )
        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            max_steps=args.max_valid_steps,
            log_interval=max(1, args.log_interval // 2),
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} psnr {train_metrics['psnr']:.2f} ssim {train_metrics['ssim']:.4f} | "
            f"valid loss {valid_metrics['loss']:.4f} psnr {valid_metrics['psnr']:.2f} ssim {valid_metrics['ssim']:.4f}"
        )
        if "feat_loss" in train_metrics:
            print(
                f"           train rec {train_metrics['rec_loss']:.4f} "
                f"feat {train_metrics['feat_loss']:.4f}"
            )
        append_log_row(log_path, epoch, train_metrics, valid_metrics)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if valid_metrics["psnr"] > best_psnr:
            best_psnr = valid_metrics["psnr"]
            torch.save(checkpoint, output_dir / "best.pt")


def append_log_row(
    log_path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
) -> None:
    row = {
        "epoch": epoch,
        "train_loss": train_metrics.get("loss", 0.0),
        "train_rec_loss": train_metrics.get("rec_loss", ""),
        "train_feat_loss": train_metrics.get("feat_loss", ""),
        "train_psnr": train_metrics.get("psnr", 0.0),
        "train_ssim": train_metrics.get("ssim", 0.0),
        "valid_loss": valid_metrics.get("loss", 0.0),
        "valid_psnr": valid_metrics.get("psnr", 0.0),
        "valid_ssim": valid_metrics.get("ssim", 0.0),
    }
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
