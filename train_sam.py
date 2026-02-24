"""
Fine-tune SAM (facebook/sam-vit-base) mask decoder on the local BUSI dataset.
Saves best checkpoint to best.pth (used by app.py).

Usage:
    python train_sam.py
"""

import glob
import os

import monai
import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SamModel, SamProcessor

DATASET_PATH = "Dataset_BUSI_with_GT"
BATCH_SIZE   = 2
NUM_EPOCHS   = 10
LR           = 1e-5
PATIENCE     = 5
SAVE_PATH    = "best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[train_sam] device = {DEVICE}")


def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0:
        return None
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    return [x_min, y_min, x_max, y_max]


class BUSISAMDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor   = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = img_path.replace(").png", ")_mask.png")

        image = Image.open(img_path).convert("RGB")
        mask  = np.array(Image.open(mask_path).convert("L").resize((256, 256)))
        mask  = (mask > 0).astype(np.uint8)

        bbox = get_bounding_box(mask)
        if bbox is None:
            # fallback: full-image box
            bbox = [0, 0, 256, 256]

        inputs = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask, dtype=torch.float32)
        return inputs


def build_dataset():
    image_paths = (
        sorted(glob.glob(os.path.join(DATASET_PATH, "benign",    "*).png")))
        + sorted(glob.glob(os.path.join(DATASET_PATH, "malignant", "*).png")))
    )
    print(f"[train_sam] Total images: {len(image_paths)}")
    split = int(len(image_paths) * 0.8)
    return image_paths[:split], image_paths[split:]


def main():
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_paths, val_paths = build_dataset()

    train_ds = BUSISAMDataset(train_paths, processor)
    val_ds   = BUSISAMDataset(val_paths,   processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    model = model.to(DEVICE)

    optimizer = Adam(model.mask_decoder.parameters(), lr=LR, weight_decay=0)
    seg_loss  = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} train"):
            batch["input_boxes"] = batch["input_boxes"].float()
            outputs = model(
                pixel_values   = batch["pixel_values"].to(DEVICE),
                input_boxes    = batch["input_boxes"].to(DEVICE),
                multimask_output = False,
            )
            pred_masks = outputs.pred_masks.squeeze(1)
            gt_masks   = batch["ground_truth_mask"].float().to(DEVICE)
            loss = seg_loss(pred_masks, gt_masks.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} val"):
                batch["input_boxes"] = batch["input_boxes"].float()
                outputs = model(
                    pixel_values   = batch["pixel_values"].to(DEVICE),
                    input_boxes    = batch["input_boxes"].to(DEVICE),
                    multimask_output = False,
                )
                pred_masks = outputs.pred_masks.squeeze(1)
                gt_masks   = batch["ground_truth_mask"].float().to(DEVICE)
                val_losses.append(seg_loss(pred_masks, gt_masks.unsqueeze(1)).item())

        train_mean = np.mean(train_losses)
        val_mean   = np.mean(val_losses)
        print(f"Epoch {epoch+1}: train_loss={train_mean:.4f}  val_loss={val_mean:.4f}")

        torch.save(model.state_dict(), "last.pth")
        if val_mean < best_loss:
            best_loss = val_mean
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved best checkpoint ({SAVE_PATH})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stop.")
                break

    print(f"Training done. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
