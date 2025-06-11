from configs.lane_det_config import CFG
from utils.logger import get_logger
from utils.metrics import tusimple_accuracy
from models.encoder import get_resnet34, freeze
from models.lane_head import LaneATTHead
from models.lane_att import LaneDetectionModel
from data.tusimple_dataset import TuSimpleDataset

import torch
from torch.utils.data import DataLoader

f_logger = get_logger("finetune")

def main():
    train_ds = TuSimpleDataset(CFG["data_root"], "train")
    val_ds = TuSimpleDataset(CFG["data_root"], "val")
    train_ld = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)

    encoder = get_resnet34(False, CFG["simclr_weights"]).to(CFG["device"])
    freeze(encoder)

    head = LaneATTHead(512, CFG["num_anchors"]).to(CFG["device"])
    model = LaneDetectionModel(encoder, head, freeze_encoder=True).to(CFG["device"])

    optim = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()

    for epoch in range(CFG["num_epochs"]):
        model.train()
        for imgs, gts in train_ld:
            imgs, gts = imgs.to(CFG["device"]), gts.to(CFG["device"])
            pred = model(imgs)
            loss = crit(pred[..., 0], gts)
            optim.zero_grad(); loss.backward(); optim.step()

        model.eval(); acc = 0.0
        with torch.no_grad():
            for imgs, gts in val_ld:
                imgs, gts = imgs.to(CFG["device"]), gts.to(CFG["device"])
                pred = model(imgs)
                acc += tusimple_accuracy(pred, gts)
        acc /= len(val_ld)
        f_logger.info(f"epoch {epoch} val‑acc {acc:.4f}")

    torch.save(model.state_dict(), "outputs/finetuned_laneatt.pt")
    f_logger.info("Saved fine‑tuned weights → outputs/finetuned_laneatt.pt")


if __name__ == "__main__":
    main()