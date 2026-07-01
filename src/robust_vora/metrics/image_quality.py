from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    prediction = prediction.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    mse = (prediction - target).square().flatten(1).mean(dim=1).clamp_min(eps)
    return 20.0 * torch.log10(torch.tensor(data_range, device=prediction.device)) - 10.0 * torch.log10(mse)


def batch_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    prediction = prediction.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction.device, prediction.dtype)

    mu_x = F.conv2d(prediction, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(prediction * prediction, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=window_size // 2, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size).expand(channels, 1, window_size, window_size)
