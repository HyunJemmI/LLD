import torch.nn as nn
import torch.nn.functional as F


class LaneATTHead(nn.Module):
    """Minimal anchor‑based head (offset + confidence per anchor)."""

    def __init__(self, in_channels: int, num_anchors: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, num_anchors * 2)  # [offset, conf] × anchors

    def forward(self, feat):
        x = self.gap(feat).flatten(1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out.view(out.size(0), -1, 2)  # (B, A, 2)