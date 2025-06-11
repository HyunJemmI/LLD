import os
import json
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class TuSimpleDataset(Dataset):
    """Minimal TuSimple loader (image → RGB tensor, label → lane points tensor)."""

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = root
        self.split = split  # train / val / test
        self.transform = transform
        self.samples = self._load_annotations()

    # -----------------------------------------------------------------
    def _load_annotations(self) -> List[Tuple[str, list, list]]:
        label_file = os.path.join(self.root, f"label_data_{self.split}.json")
        with open(label_file) as f:
            data = json.load(f)
        samples = []
        for entry in data:
            img_path = os.path.join(self.root, "clips", entry["raw_file"])
            samples.append((img_path, entry["lanes"], entry["h_samples"]))
        return samples

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lanes, h_samples = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        target = self._encode_lanes(lanes, h_samples)
        return image, target

    # -----------------------------------------------------------------
    @staticmethod
    def _encode_lanes(lanes: list, h_samples: list):
        """Pack first 4 lane x‑coords into a tensor. (Toy encoder)"""
        import numpy as np
        padded = np.zeros((4, len(h_samples)), dtype=np.float32)
        for i, lane in enumerate(lanes[:4]):
            padded[i, :] = lane
        return torch.tensor(padded)