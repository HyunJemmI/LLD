import torch


def tusimple_accuracy(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 5.0):
    """Very rough placeholder for TuSimple metric (x‑error < thresh)."""
    # pred: (B, A, 2) – using only offset channel here (column 0)
    pred_x = pred[..., 0]
    diff = (pred_x - gt).abs()
    correct = (diff < thresh).float().mean()
    return correct.item()