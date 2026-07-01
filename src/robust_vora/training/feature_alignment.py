from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_FEATURE_LAYERS = (
    "model.encoder_level1",
    "model.encoder_level2",
    "model.encoder_level3",
    "model.latent",
)


def extract_features(
    model: nn.Module,
    image: torch.Tensor,
    layer_names: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
    detach: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())

    for layer_name in layer_names:
        if layer_name not in modules:
            raise KeyError(f"Feature layer not found: {layer_name}")
        module = modules[layer_name]

        def hook(_module, _inputs, output, name=layer_name):
            features[name] = output.detach() if detach else output

        handles.append(module.register_forward_hook(hook))

    output = model(image)
    for handle in handles:
        handle.remove()
    return features, output


def feature_alignment_loss(
    source_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, source in source_features.items():
        target = target_features[name]
        source_norm = F.normalize(source.flatten(2), dim=1)
        target_norm = F.normalize(target.flatten(2), dim=1)
        losses.append((source_norm - target_norm).square().mean())
    return torch.stack(losses).mean()
