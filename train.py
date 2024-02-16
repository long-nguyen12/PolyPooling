import argparse
import os
import time
from pathlib import Path

import albumentations as A
import torch
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.datasets.polyp import PolypDB
from models.models import *
from models.optimizers import get_optimizer
from models.schedulers import get_scheduler
from models.utils.utils import AvgMeter, clip_gradient, fix_seeds, setup_cudnn
from val import evaluate

batch_size = 8


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def main(cfg, save_dir, train_loader, val_loader):
    start = time.time()
    best_mIoU = 0.0
    device = torch.device(cfg["DEVICE"])
    train_cfg = cfg["TRAIN"]
    model_cfg = cfg["MODEL"]
    optim_cfg, sched_cfg = cfg["OPTIMIZER"], cfg["SCHEDULER"]
    epochs, lr = train_cfg["EPOCHS"], optim_cfg["LR"]

    model = eval(model_cfg["NAME"])(model_cfg["BACKBONE"], 2)
    model.init_pretrained(model_cfg["PRETRAINED"])
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters {model_cfg['NAME']}: {total_params}")

    writer = SummaryWriter(str(save_dir / "logs"))
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25]

    iters_per_epoch = len(train_loader.dataset) // batch_size

    optimizer = get_optimizer(model, optim_cfg["NAME"], lr, optim_cfg["WEIGHT_DECAY"])
    scheduler = get_scheduler(
        sched_cfg["NAME"],
        optimizer,
        epochs * iters_per_epoch,
        sched_cfg["POWER"],
        iters_per_epoch * sched_cfg["WARMUP"],
        sched_cfg["WARMUP_RATIO"],
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader),
            total=iters_per_epoch,
            desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}",
        )

        for iter, (img, lbl) in pbar:
            if epoch <= 1:
                optimizer.param_groups[0]["lr"] = (
                    (epoch * iter) / (1.0 * iters_per_epoch) * lr
                )
            else:
                scheduler.step()

            for rate in size_rates:
                optimizer.zero_grad(set_to_none=True)

                img = img.to(device)
                lbl = lbl.to(device)

                trainsize = int(352 * rate)
                if rate != 1:
                    if lbl.shape[1] != 1:
                        lbl = lbl.unsqueeze(1).float()
                    img = F.interpolate(
                        img,
                        size=(trainsize, trainsize),
                        mode="bicubic",
                        align_corners=True,
                    )
                    lbl = F.interpolate(
                        lbl,
                        size=(trainsize, trainsize),
                        mode="bicubic",
                        align_corners=True,
                    )

                logits, score4, score3, score2, score1 = model(img)
                loss_0 = structure_loss(logits, lbl)
                loss_4 = structure_loss(score4, lbl)
                loss_3 = structure_loss(score3, lbl)
                loss_2 = structure_loss(score2, lbl)
                loss_1 = structure_loss(score1, lbl)
                loss = loss_0 + loss_2 + loss_3 + loss_4 + loss_1

                loss.backward()
                clip_gradient(optimizer, 0.5)
                optimizer.step()

                if rate == 1:
                    loss_record.update(loss.data, train_cfg["BATCH_SIZE"])

            torch.cuda.synchronize()

            train_loss += loss.data

            pbar.set_description(
                f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}"
            )

        train_loss /= iter + 1
        writer.add_scalar("train/loss", train_loss, epoch)
        torch.cuda.empty_cache()

        if epoch % train_cfg["EVAL_INTERVAL"] == 0 or epoch == epochs:
            miou = evaluate(model, val_loader, device, "Training")[0]

            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.state_dict(), save_dir / "best.pth")
            torch.save(model.state_dict(), save_dir / f"checkpoint{epoch}.pth")

            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)
    print(time.strftime("%H:%M:%S", end))


def create_dataloaders(dir, image_size, batch_size, num_workers=os.cpu_count()):
    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = PolypDB(root=dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader, dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/custom.yaml",
        help="Configuration file to use",
    )
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    save_dir = Path(cfg["SAVE_DIR"])
    save_dir.mkdir(exist_ok=True)

    dataloader, dataset = create_dataloaders(
        "data/TrainDataset/", [352, 352], batch_size
    )

    train_ratio = 0.8
    val_ratio = 0.2
    num_samples = len(dataloader.dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_val_samples = num_samples - num_train_samples
    train_set, val_set = torch.utils.data.random_split(
        dataset, [num_train_samples, num_val_samples]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    main(cfg, save_dir, train_loader, val_loader)
