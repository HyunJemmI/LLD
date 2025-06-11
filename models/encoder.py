import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


def get_resnet34(pretrained: bool = False, simclr_weights: str | None = None) -> nn.Module:
    """Return ResNet‑34 feature extractor.

    Args:
        pretrained (bool): If True, load ImageNet weights.
        simclr_weights (str | None): Optional path to SimCLR‑pretrained weights.
    """
    backbone = models.resnet34(pretrained=pretrained)
    # Drop the average‑pool & fc layers → CxHxW feature map (C=512)
    encoder = nn.Sequential(*list(backbone.children())[:-2])

    if simclr_weights:
        state = torch.load(simclr_weights, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[encoder] Loaded SimCLR weights → missing={len(missing)}, unexpected={len(unexpected)}")

    return encoder


def freeze(module: nn.Module):
    """Freeze parameters & put BatchNorms in eval mode."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()