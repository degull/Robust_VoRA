from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


class FrozenRestormer(nn.Module):
    """Thin wrapper around the official Restormer implementation."""

    def __init__(
        self,
        restormer_root: str | Path = "third_party/Restormer",
        checkpoint: str | Path | None = None,
        freeze: bool = True,
        **restormer_kwargs: Any,
    ) -> None:
        super().__init__()
        Restormer = _load_official_restormer(restormer_root)
        self.model = Restormer(**restormer_kwargs)

        if checkpoint:
            self.load_checkpoint(checkpoint)

        if freeze:
            self.freeze()

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        height, width = degraded.shape[-2:]
        padded = _pad_to_multiple(degraded, multiple=8)
        restored = self.model(padded)
        restored = restored[..., :height, :width]
        return restored.clamp(0.0, 1.0)

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict):
            for key in ("params", "state_dict", "model", "net"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        state = _strip_prefixes(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"Restormer checkpoint missing keys: {len(missing)}")
        if unexpected:
            print(f"Restormer checkpoint unexpected keys: {len(unexpected)}")


def create_restormer(
    restormer_root: str | Path = "third_party/Restormer",
    checkpoint: str | Path | None = None,
    freeze: bool = True,
    layer_norm_type: str = "BiasFree",
) -> FrozenRestormer:
    return FrozenRestormer(
        restormer_root=restormer_root,
        checkpoint=checkpoint,
        freeze=freeze,
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


def _load_official_restormer(restormer_root: str | Path):
    root = Path(restormer_root).resolve()
    arch_dir = root / "basicsr" / "models" / "archs"
    arch_file = arch_dir / "restormer_arch.py"
    if not arch_file.exists():
        raise FileNotFoundError(
            f"Official Restormer architecture not found at {arch_file}. "
            "Clone https://github.com/swz30/Restormer into third_party/Restormer."
        )
    sys.path.insert(0, str(arch_dir))
    from restormer_arch import Restormer

    return Restormer


def _strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    stripped = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        stripped[new_key] = value
    return stripped


def _pad_to_multiple(image: torch.Tensor, multiple: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image
    return torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
