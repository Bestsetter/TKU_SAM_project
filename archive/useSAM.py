import numpy as np 
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import SamProcessor
import torch
from transformers import SamModel 
import random

def get_bounding_box(ground_truth_map):
      # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


model = SamModel.from_pretrained("facebook/sam-vit-base")
model.load_state_dict(torch.load("best.pth"))
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

dataset = load_dataset("nielsr/breast-cancer", split="train")
# load image
idx = random.randint(1, 100)
example  = dataset[idx]
image = example["image"]
# image.show()

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to('cuda') 
# prepare image + box prompt for the model
inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
for k,v in inputs.items():
  print(k,v.shape)

model.eval()

# forward pass
with torch.no_grad():
  outputs = model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)



# fig, axes = plt.subplots()
# axes.imshow(np.array(image))
# show_mask(medsam_seg, axes)
# axes.title.set_text(f"Predicted mask")
# axes.axis("off")
# plt.show()

fig, axes = plt.subplots(1, 3)
# 在子圖 1 中顯示第一張圖片
axes[0].imshow(image)
axes[0].set_title("origin Image")

# 在子圖 2 中顯示第二張圖片

axes[1].imshow(np.array(image))
show_mask(medsam_seg, axes[1])
axes[1].title.set_text(f"Predicted mask")

axes[2].imshow(np.array(image))
ground_truth_seg = np.array(example["label"])
show_mask(ground_truth_seg, axes[2])
axes[2].title.set_text(f"Ground truth mask")

# 顯示子圖
plt.show()