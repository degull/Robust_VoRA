from __future__ import annotations

import time
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from robust_vora.metrics import batch_psnr, batch_ssim
from robust_vora.training.feature_alignment import DEFAULT_FEATURE_LAYERS, extract_features, feature_alignment_loss


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int | None = None,
    log_interval: int = 100,
    epoch: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    steps = 0

    expected_steps = _expected_steps(loader, max_steps)
    start_time = time.time()
    for batch in loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        restored = model(degraded)
        loss = F.l1_loss(restored, clean)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_psnr += batch_psnr(restored, clean).mean().item()
            total_ssim += batch_ssim(restored, clean).mean().item()
            steps += 1

        if max_steps is not None and steps >= max_steps:
            break
        if log_interval > 0 and steps % log_interval == 0:
            _print_progress("train", epoch, steps, expected_steps, start_time, total_loss, total_psnr, total_ssim)

    return _average_metrics(total_loss, total_psnr, total_ssim, steps)


def train_one_epoch_with_feature_alignment(
    model: nn.Module,
    clean_feature_model: nn.Module,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_feat: float,
    feature_layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
    max_steps: int | None = None,
    log_interval: int = 100,
    epoch: int | None = None,
) -> dict[str, float]:
    model.train()
    clean_feature_model.eval()
    total_loss = 0.0
    total_rec = 0.0
    total_feat = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    steps = 0

    expected_steps = _expected_steps(loader, max_steps)
    start_time = time.time()
    for batch in loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        with torch.no_grad():
            clean_features, _ = extract_features(
                clean_feature_model,
                clean,
                layer_names=feature_layers,
                detach=True,
            )

        adapted_features, restored = extract_features(
            model,
            degraded,
            layer_names=feature_layers,
            detach=False,
        )
        rec_loss = F.l1_loss(restored, clean)
        feat_loss = feature_alignment_loss(adapted_features, clean_features)
        loss = rec_loss + lambda_feat * feat_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_feat += feat_loss.item()
            total_psnr += batch_psnr(restored, clean).mean().item()
            total_ssim += batch_ssim(restored, clean).mean().item()
            steps += 1

        if max_steps is not None and steps >= max_steps:
            break
        if log_interval > 0 and steps % log_interval == 0:
            _print_progress(
                "train",
                epoch,
                steps,
                expected_steps,
                start_time,
                total_loss,
                total_psnr,
                total_ssim,
                extra=f"rec {total_rec / max(steps, 1):.4f} feat {total_feat / max(steps, 1):.4f}",
            )

    metrics = _average_metrics(total_loss, total_psnr, total_ssim, steps)
    divisor = max(steps, 1)
    metrics["rec_loss"] = total_rec / divisor
    metrics["feat_loss"] = total_feat / divisor
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict],
    device: torch.device,
    max_steps: int | None = None,
    log_interval: int = 50,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    steps = 0

    expected_steps = _expected_steps(loader, max_steps)
    start_time = time.time()
    for batch in loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        restored = model(degraded)
        loss = F.l1_loss(restored, clean)

        total_loss += loss.item()
        total_psnr += batch_psnr(restored, clean).mean().item()
        total_ssim += batch_ssim(restored, clean).mean().item()
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break
        if log_interval > 0 and steps % log_interval == 0:
            _print_progress("valid", None, steps, expected_steps, start_time, total_loss, total_psnr, total_ssim)

    return _average_metrics(total_loss, total_psnr, total_ssim, steps)


def _average_metrics(total_loss: float, total_psnr: float, total_ssim: float, steps: int) -> dict[str, float]:
    divisor = max(steps, 1)
    return {
        "loss": total_loss / divisor,
        "psnr": total_psnr / divisor,
        "ssim": total_ssim / divisor,
        "steps": float(steps),
    }


def _expected_steps(loader: Iterable[dict], max_steps: int | None) -> int | None:
    try:
        loader_len = len(loader)  # type: ignore[arg-type]
    except TypeError:
        loader_len = None
    if max_steps is None:
        return loader_len
    return min(max_steps, loader_len) if loader_len is not None else max_steps


def _print_progress(
    phase: str,
    epoch: int | None,
    steps: int,
    expected_steps: int | None,
    start_time: float,
    total_loss: float,
    total_psnr: float,
    total_ssim: float,
    extra: str = "",
) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    speed = steps / elapsed
    if expected_steps is None:
        progress = f"{steps}"
        eta = "--:--"
    else:
        progress = f"{steps}/{expected_steps}"
        remaining = max(expected_steps - steps, 0) / max(speed, 1e-6)
        eta = _format_seconds(remaining)
    prefix = f"[epoch {epoch:03d}] " if epoch is not None else ""
    message = (
        f"{prefix}{phase} {progress} | "
        f"loss {total_loss / max(steps, 1):.4f} "
        f"psnr {total_psnr / max(steps, 1):.2f} "
        f"ssim {total_ssim / max(steps, 1):.4f} | "
        f"{speed:.2f} step/s eta {eta}"
    )
    if extra:
        message += f" | {extra}"
    print(message, flush=True)


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
