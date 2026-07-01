from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .restormer import FrozenRestormer


class PerturbationEncoder(nn.Module):
    """Encodes simple image statistics into a perturbation code."""

    def __init__(self, code_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean(dim=(2, 3))
        std = image.std(dim=(2, 3), unbiased=False)
        brightness = image.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        contrast = image.std(dim=(1, 2, 3), unbiased=False, keepdim=False).unsqueeze(1)
        stats = torch.cat([mean, std, brightness, contrast], dim=1)
        return self.net(stats)


class VoRAAdapter(nn.Module):
    """Perturbation-conditioned low-rank adapter with a quadratic Volterra branch."""

    def __init__(self, channels: int, rank: int = 8, code_dim: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.rank = rank
        self.scale = scale
        self.lora_scaling = 1.0 / rank
        self.volterra_scaling = 1.0 / rank
        self.norm = nn.GroupNorm(1, channels)
        self.A_l = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        self.A_v = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.B_v = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        self.lora_gate = nn.Linear(code_dim, rank)
        self.volterra_gate = nn.Linear(code_dim, rank)
        self.channel_gate = nn.Linear(code_dim, channels)
        self.condition: torch.Tensor | None = None

        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)
        nn.init.kaiming_uniform_(self.A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_v.weight)
        nn.init.zeros_(self.lora_gate.weight)
        nn.init.zeros_(self.lora_gate.bias)
        nn.init.zeros_(self.volterra_gate.weight)
        nn.init.zeros_(self.volterra_gate.bias)
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.zeros_(self.channel_gate.bias)

    def set_condition(self, condition: torch.Tensor) -> None:
        self.condition = condition

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if self.condition is None:
            raise RuntimeError("VoRAAdapter condition was not set before forward.")
        lora_gate = torch.sigmoid(self.lora_gate(self.condition)).view(feature.shape[0], self.rank, 1, 1)
        volterra_gate = torch.sigmoid(self.volterra_gate(self.condition)).view(feature.shape[0], self.rank, 1, 1)
        channel_gate = torch.sigmoid(self.channel_gate(self.condition)).view(feature.shape[0], feature.shape[1], 1, 1)

        normalized = self.norm(feature)
        lora_hidden = self.A_l(normalized) * lora_gate
        lora_update = self.B_l(lora_hidden) * self.lora_scaling

        volterra_hidden = self.A_v(normalized)
        volterra_hidden = (volterra_hidden * volterra_hidden) * volterra_gate
        volterra_update = self.B_v(volterra_hidden) * self.volterra_scaling

        update = (lora_update + volterra_update) * channel_gate
        return feature + self.scale * update


class LoRAAdapter(nn.Module):
    """Perturbation-conditioned LoRA branch without Volterra interaction."""

    def __init__(self, channels: int, rank: int = 8, code_dim: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.rank = rank
        self.scale = scale
        self.lora_scaling = 1.0 / rank
        self.norm = nn.GroupNorm(1, channels)
        self.A_l = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        self.lora_gate = nn.Linear(code_dim, rank)
        self.channel_gate = nn.Linear(code_dim, channels)
        self.condition: torch.Tensor | None = None

        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)
        nn.init.zeros_(self.lora_gate.weight)
        nn.init.zeros_(self.lora_gate.bias)
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.zeros_(self.channel_gate.bias)

    def set_condition(self, condition: torch.Tensor) -> None:
        self.condition = condition

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if self.condition is None:
            raise RuntimeError("LoRAAdapter condition was not set before forward.")
        lora_gate = torch.sigmoid(self.lora_gate(self.condition)).view(feature.shape[0], self.rank, 1, 1)
        channel_gate = torch.sigmoid(self.channel_gate(self.condition)).view(feature.shape[0], feature.shape[1], 1, 1)
        hidden = self.A_l(self.norm(feature)) * lora_gate
        update = self.B_l(hidden) * self.lora_scaling
        update = update * channel_gate
        return feature + self.scale * update


class VoRAWrappedBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        channels: int,
        rank: int,
        code_dim: int,
        scale: float,
        adapter_type: str = "vora",
    ) -> None:
        super().__init__()
        self.block = block
        if adapter_type == "lora":
            self.adapter = LoRAAdapter(channels=channels, rank=rank, code_dim=code_dim, scale=scale)
        elif adapter_type == "vora":
            self.adapter = VoRAAdapter(channels=channels, rank=rank, code_dim=code_dim, scale=scale)
        else:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}")

    def set_condition(self, condition: torch.Tensor) -> None:
        self.adapter.set_condition(condition)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.block(feature))


class AdaptiveVoRARestormer(FrozenRestormer):
    """Frozen Restormer backbone with trainable Adaptive VoRA adapters."""

    def __init__(
        self,
        restormer_root: str | Path = "third_party/Restormer",
        checkpoint: str | Path | None = None,
        layer_norm_type: str = "WithBias",
        rank: int = 8,
        code_dim: int = 64,
        adapter_scale: float = 1.0,
        adapter_type: str = "vora",
    ) -> None:
        super().__init__(
            restormer_root=restormer_root,
            checkpoint=checkpoint,
            freeze=True,
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type=layer_norm_type,
            dual_pixel_task=False,
        )
        self.perturbation_encoder = PerturbationEncoder(code_dim=code_dim)
        self.vora_adapters = nn.ModuleList()
        self._inject_adapters(rank=rank, code_dim=code_dim, scale=adapter_scale, adapter_type=adapter_type)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        condition = self.perturbation_encoder(degraded)
        for adapter in self.vora_adapters:
            adapter.set_condition(condition)
        return super().forward(degraded)

    def _inject_adapters(self, rank: int, code_dim: int, scale: float, adapter_type: str) -> None:
        specs = (
            ("encoder_level1", 48),
            ("encoder_level2", 96),
            ("encoder_level3", 192),
            ("latent", 384),
            ("decoder_level3", 192),
            ("decoder_level2", 96),
            ("decoder_level1", 96),
            ("refinement", 96),
        )
        for module_name, channels in specs:
            sequence = getattr(self.model, module_name)
            wrapped = []
            for block in sequence:
                wrapper = VoRAWrappedBlock(
                    block=block,
                    channels=channels,
                    rank=rank,
                    code_dim=code_dim,
                    scale=scale,
                    adapter_type=adapter_type,
                )
                wrapped.append(wrapper)
                self.vora_adapters.append(wrapper)
            setattr(self.model, module_name, nn.Sequential(*wrapped))


def create_adaptive_vora_restormer(
    restormer_root: str | Path = "third_party/Restormer",
    checkpoint: str | Path | None = None,
    layer_norm_type: str = "WithBias",
    rank: int = 8,
    code_dim: int = 64,
    adapter_scale: float = 1.0,
    adapter_type: str = "vora",
) -> AdaptiveVoRARestormer:
    return AdaptiveVoRARestormer(
        restormer_root=restormer_root,
        checkpoint=checkpoint,
        layer_norm_type=layer_norm_type,
        rank=rank,
        code_dim=code_dim,
        adapter_scale=adapter_scale,
        adapter_type=adapter_type,
    )


def create_adaptive_lora_restormer(
    restormer_root: str | Path = "third_party/Restormer",
    checkpoint: str | Path | None = None,
    layer_norm_type: str = "WithBias",
    rank: int = 8,
    code_dim: int = 64,
    adapter_scale: float = 1.0,
) -> AdaptiveVoRARestormer:
    return create_adaptive_vora_restormer(
        restormer_root=restormer_root,
        checkpoint=checkpoint,
        layer_norm_type=layer_norm_type,
        rank=rank,
        code_dim=code_dim,
        adapter_scale=adapter_scale,
        adapter_type="lora",
    )
