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
from datasets import DatasetDict, load_dataset
import os

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


#載入model 架構
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.load_state_dict(torch.load("best.pth"))
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

#設定測試圖片資料夾(隨機選取圖片，normal 不適用)
folder_A = "D:/TKU_SAM_project/Dataset_BUSI_with_GT/malignant"
folder_B = "D:/TKU_SAM_project/Dataset_BUSI_with_GT/begnin"
selected_folder = random.choice([folder_A, folder_B])
image_files = [f for f in os.listdir(selected_folder) if f.endswith(").png")]
selected_image = random.choice(image_files)
image_path = os.path.join(selected_folder, selected_image)
mask_path = image_path.replace(").png", ")_mask.png")
# print(image_path)
# print(mask_path)
image = Image.open(image_path)
mask = Image.open(mask_path)
image = image.resize((256, 256))
mask = mask.resize((256, 256))
# print(image.size)
# print(mask.size)


ground_truth_mask = np.array(mask)
prompt = get_bounding_box(ground_truth_mask)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to('cuda') 
# prepare image + box prompt for the model
inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
# for k,v in inputs.items():
#   print(k,v.shape)

model.eval()

# forward pass
with torch.no_grad():
  outputs = model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)



#顯示最後結果(原圖、預測、答案)
fig, axes = plt.subplots(1, 3)
# 在子圖 1 中顯示第一張圖片
axes[0].imshow(image)
axes[0].set_title("origin Image")

axes[1].imshow(np.array(image))
show_mask(medsam_seg, axes[1])
axes[1].title.set_text(f"Predicted mask")

axes[2].imshow(np.array(image))
ground_truth_seg = np.array(mask)
show_mask(ground_truth_seg, axes[2])
axes[2].title.set_text(f"Ground truth mask")

#存檔
output_folder = "save_ans"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

save_name = os.path.basename(image_path)
save_name += "_ans.png"
output_path = os.path.join(output_folder, save_name)
plt.savefig(output_path)

plt.show()