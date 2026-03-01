"""
Evaluate SAM (facebook/sam-vit-base) in ZERO-SHOT mode on BUSI val set.
Uses the same 80/20 split as train_sam.py — NO weights from best.pth are loaded.
Bounding box is derived from GT mask (simulating a tight user-drawn box).

Usage:
    python eval_sam_zeroshot.py
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor

DATASET_PATH = "Dataset_BUSI_with_GT"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[eval_sam_zeroshot] device = {DEVICE}")


def get_bounding_box(ground_truth_map):
    """Tight bounding box from GT mask (no random jitter for fair eval)."""
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0:
        return [0, 0, 256, 256]
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    return [x_min, y_min, x_max, y_max]


def seg_metrics(pred, gt):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    return precision, recall, dice, iou


def build_val_paths():
    image_paths = (
        sorted(glob.glob(os.path.join(DATASET_PATH, "benign",    "*).png")))
        + sorted(glob.glob(os.path.join(DATASET_PATH, "malignant", "*).png")))
    )
    split = int(len(image_paths) * 0.8)
    val_paths = image_paths[split:]
    print(f"[eval_sam_zeroshot] Val images: {len(val_paths)}")
    return val_paths


def main():
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    # Zero-shot: load pretrained weights only, no best.pth
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    model.eval()

    val_paths = build_val_paths()

    all_precision, all_recall, all_dice, all_iou = [], [], [], []
    skipped = 0

    for img_path in tqdm(val_paths, desc="Zero-shot eval"):
        mask_path = img_path.replace(").png", ")_mask.png")
        if not os.path.exists(mask_path):
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask  = np.array(Image.open(mask_path).convert("L").resize((256, 256)))
        gt    = (mask > 0).astype(np.int32)

        if gt.sum() == 0:
            skipped += 1
            continue

        bbox = get_bounding_box(gt)
        inputs = processor(image, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        pred = (torch.sigmoid(outputs.pred_masks.squeeze(1))
                .cpu().numpy().squeeze() > 0.5).astype(np.int32)

        p, r, d, iou = seg_metrics(pred, gt)
        all_precision.append(p)
        all_recall.append(r)
        all_dice.append(d)
        all_iou.append(iou)

    print(f"\n{'='*45}")
    print(f"  SAM Zero-Shot Results on BUSI Val Set")
    print(f"{'='*45}")
    print(f"  Samples evaluated : {len(all_dice)}  (skipped: {skipped})")
    print(f"  Mean Dice         : {np.mean(all_dice):.4f}")
    print(f"  Mean IoU          : {np.mean(all_iou):.4f}")
    print(f"  Mean Precision    : {np.mean(all_precision):.4f}")
    print(f"  Mean Recall       : {np.mean(all_recall):.4f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
