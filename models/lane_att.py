import torch.nn as nn


class LaneDetectionModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.encoder(x)
        out = self.head(feat)
        return out