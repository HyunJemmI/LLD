import torch

CFG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,
    "num_epochs": 30,
    "num_anchors": 4,
    "data_root": "/home1/hyunje0/LLD/dataset/contrastive_test_set_aug",
    "simclr_weights": "outputs/simclr_encoder.pt",
}