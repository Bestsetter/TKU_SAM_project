"""
Train UnetPlusPlus (ResNet34 encoder) on the local BUSI dataset.
Saves best checkpoint to unetplusplus_chkpt/unetplusplus.pth (used by app.py).

Usage:
    python train_unet.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import segmentation_models_pytorch as smp

DATASET_PATH    = "Dataset_BUSI_with_GT"
CHECKPOINT_DIR  = "unetplusplus_chkpt"
CHECKPOINT_NAME = "unetplusplus"
BATCH_SIZE      = 8
EPOCHS          = 30
LR              = 3e-4
PATIENCE        = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[train_unet] device = {DEVICE}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BUSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_size=(256, 256)):
        self.df         = df
        self.input_size = input_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = cv2.imread(row["images"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row["masks"],  cv2.IMREAD_GRAYSCALE)

        img  = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        resize = transforms.Resize(self.input_size, antialias=True)
        return resize(img), resize(mask)


def build_dataframe():
    rows = []
    for cls in ("benign", "malignant"):
        cls_path = os.path.join(DATASET_PATH, cls)
        images   = sorted(glob.glob(os.path.join(cls_path, "*).png")))
        for img_path in images:
            mask_path = img_path.replace(").png", ")_mask.png")
            if os.path.exists(mask_path):
                rows.append({"images": img_path, "masks": mask_path})
    df = pd.DataFrame(rows)
    print(f"[train_unet] Total pairs: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_coefficient(preds, targets, smooth=1.0):
    assert preds.size() == targets.size()
    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    return (2.0 * (iflat * tflat).sum() + smooth) / (iflat.sum() + tflat.sum() + smooth)


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, bce_loss, device, is_train, desc):
    model.train() if is_train else model.eval()
    losses, dices = [], []
    progress = tqdm(loader, desc=desc, leave=False)
    threshold = nn.Threshold(0.5, 0.0)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for image, mask in progress:
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            pred_sig = torch.sigmoid(pred)

            dice = dice_coefficient(threshold(pred_sig), mask)
            loss = 0.8 * (1 - dice) + 0.2 * bce_loss(pred_sig, mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.detach().item())
            dices.append(dice.detach().item())
            progress.set_postfix(loss=f"{np.mean(losses):.4f}", dice=f"{np.mean(dices):.4f}")

    return np.mean(losses), np.mean(dices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = build_dataframe()
    train_df, val_df = train_test_split(df, train_size=0.8, random_state=42)

    train_loader = DataLoader(BUSIDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(BUSIDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1)
    bce_loss  = nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME + ".pth")

    best_val_loss = float("inf")
    best_epoch    = 0

    for epoch in range(EPOCHS):
        train_loss, train_dice = run_epoch(
            model, train_loader, optimizer, bce_loss, DEVICE, is_train=True,
            desc=f"Epoch {epoch+1}/{EPOCHS} train"
        )
        val_loss, val_dice = run_epoch(
            model, val_loader, optimizer, bce_loss, DEVICE, is_train=False,
            desc=f"Epoch {epoch+1}/{EPOCHS} val"
        )
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch+1:03d} | "
            f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"val loss={val_loss:.4f} dice={val_dice:.4f}"
        )

        if val_loss < best_val_loss - 0.01:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best checkpoint ({save_path})")
        elif epoch - best_epoch >= PATIENCE:
            print("Early stop.")
            break

    print(f"Training done. Best val loss: {best_val_loss:.4f}  saved to {save_path}")


if __name__ == "__main__":
    main()
