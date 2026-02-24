"""Train the ResNet CNN classifier on BUSI (benign vs malignant)."""
import os
import glob

import cv2 as cv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from cnn_classifier import ResNetClassifier

BUSI_PATH  = "Dataset_BUSI_with_GT"
SAVE_DIR   = "cnn_chkpt"
EPOCHS     = 200
BATCH_SIZE = 32
LR         = 1e-4
PATIENCE   = 20


class BUSIClassifDataset(Dataset):
    def __init__(self, df, input_size=(256, 256)):
        self.df = df
        self.resize = transforms.Resize(input_size, antialias=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv.imread(row["images"], cv.IMREAD_GRAYSCALE)
        img = self.resize(TF.to_tensor(img))
        label = 0 if "benign" in row["images"] else 1   # 0=benign, 1=malignant
        return img, label


def build_df():
    rows = []
    for cls in ("benign", "malignant"):
        folder = os.path.join(BUSI_PATH, cls)
        for p in sorted(glob.glob(folder + "/*).png")):
            rows.append({"images": p, "label": cls})
    return pd.DataFrame(rows)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_cnn] device = {device}")

    df = build_df()
    n_benign    = sum(df.label == "benign")
    n_malignant = sum(df.label == "malignant")
    print(f"[train_cnn] dataset: {len(df)} total  (benign={n_benign}, malignant={n_malignant})")

    dataset = BUSIClassifDataset(df)
    n_val   = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # --- Fix 1: WeightedRandomSampler (oversample minority class) ---
    train_labels = [dataset[i][1] for i in train_ds.indices]
    class_counts = [train_labels.count(0), train_labels.count(1)]   # [benign, malignant]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,   drop_last=False)

    model = ResNetClassifier(in_channel=1, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- Fix 2: Weighted CrossEntropyLoss (penalise malignant misclassification more) ---
    total = n_benign + n_malignant
    w = torch.tensor([total / (2 * n_benign), total / (2 * n_malignant)],
                     dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    print(f"[train_cnn] class weights: benign={w[0]:.3f}, malignant={w[1]:.3f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs   = imgs.to(device, torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, torch.float32)
                preds = model(imgs).argmax(dim=1).cpu().tolist()
                all_preds  += preds
                all_labels += labels.tolist()
        acc = accuracy_score(all_labels, all_preds)
        print(f"  loss={avg_loss:.4f}  val_acc={acc:.4f}")

        # Per-class breakdown every 5 epochs
        if epoch % 5 == 0:
            print(classification_report(all_labels, all_preds,
                                        target_names=["benign", "malignant"], zero_division=0))

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cnn.pth"))
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cnn_best.pth"))
            print(f"  [best] saved  acc={best_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stop.")
                break

    print(f"\n[train_cnn] Done. Best val acc = {best_acc:.4f}")
    print(f"[train_cnn] Weights saved to {SAVE_DIR}/cnn_best.pth")


if __name__ == "__main__":
    train()
