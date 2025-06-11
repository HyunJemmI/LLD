from configs.lane_det_config import CFG
from utils.logger import get_logger
from models.encoder import get_resnet34

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# 3rd‑party: lightly – install via `pip install lightly`
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.data import LightlyDataset

logger = get_logger("contrastive")


def get_transform():
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomApply([T.ColorJitter(.8, .8, .8, .2)], p=.8),
        T.RandomGrayscale(p=.2),
        T.GaussianBlur(3),
        T.ToTensor(),
    ])


def main():
    dataset = LightlyDataset(input_dir=CFG["data_root"], transform=get_transform())
    loader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=4, drop_last=True)

    encoder = get_resnet34(pretrained=False).to(CFG["device"])
    projector = SimCLRProjectionHead(512, 512, 128).to(CFG["device"])
    model = torch.nn.Sequential(encoder, projector)

    loss_fn = NTXentLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(CFG["num_epochs"]):
        model.train()
        for (view1, view2), _ in loader:
            view1, view2 = view1.to(CFG["device"]), view2.to(CFG["device"])
            z1 = model(view1)
            z2 = model(view2)
            loss = loss_fn(z1, z2)
            optim.zero_grad(); loss.backward(); optim.step()
        logger.info(f"epoch {epoch} contrastive‑loss {loss.item():.4f}")

    torch.save(encoder.state_dict(), CFG["simclr_weights"])
    logger.info("Saved encoder → %s", CFG["simclr_weights"])


if __name__ == "__main__":
    main()