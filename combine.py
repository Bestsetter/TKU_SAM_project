import os
import pandas as pd
import glob
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset
from transformers import SamProcessor
from transformers import SamModel 
import segmentation_models_pytorch as smp
from PIL import Image

device = 'cuda'if torch.cuda.is_available() else 'cpu'

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


def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()
    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice

def dice_score(mask1, mask2):
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(mask1) + np.sum(mask2)
    dice = 2.0 * intersection / union
    return dice

dataset_paths = ['Dataset_BUSI_with_GT/benign/', 'Dataset_BUSI_with_GT/malignant']
dataset_path = random.choice(dataset_paths)
image_path = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(').png')]
image_path = random.choice(image_path)
print(image_path)
mask_path = image_path.replace('.png', '_mask.png')
print(mask_path)
save_path = os.path.relpath(image_path, dataset_path)
print(save_path)

# Unet++ model start
model_unetpp = smp.UnetPlusPlus(encoder_name="resnet34",
                        encoder_weights=None,
                        in_channels=1,
                        classes=1,
                        ).to(device)

model_unetpp.load_state_dict(torch.load('unetplusplus_chkpt/unetplusplus.pth'))

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

oimage = Image.open(image_path)
image = Image.open(image_path).convert('L')
mask = Image.open(mask_path)
image = image.resize((256, 256))
oimage = oimage.resize((256, 256))
mask = mask.resize((256, 256))

image_tensor = preprocess(image).unsqueeze(0).to(device)
mask_tensor = preprocess(mask).unsqueeze(0).to(device)

with torch.no_grad():
    model_unetpp.eval()
    pred_mask = model_unetpp(image_tensor)
    pred_mask = (torch.sigmoid(pred_mask) > 0.7).int()
    dice = dice_coefficient(pred_mask,mask_tensor).cpu().item()

# SAM model start
model_SAM = SamModel.from_pretrained("facebook/sam-vit-base")
model_SAM.load_state_dict(torch.load("best.pth"))
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
prompt = get_bounding_box(np.array(mask))
model_SAM = model_SAM.to('cuda') 
inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)

with torch.no_grad():
    model_SAM.eval()
    outputs = model_SAM(**inputs, multimask_output=False)
    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


# final show

fig, axs = plt.subplots(1,4)
axs[0].imshow(oimage)
axs[0].set_title('Image')
axs[0].axis('off')
axs[1].imshow(mask)
axs[1].set_title('Mask')
axs[1].axis('off')

axs[2].imshow(medsam_seg, cmap='gray')
axs[2].set_title(f"SAM's\npredict_mask\ndice = {dice_score(mask, medsam_seg) :.2f}")
axs[2].axis('off')



# 原藍遮罩
# axs[3].imshow(np.array(oimage))
# show_mask(medsam_seg, axs[3])
axs[3].imshow(pred_mask[0, 0].cpu().numpy(), cmap='gray')  # 取第一張圖的第一個通道的內容
axs[3].set_title(f"Unet++'s\npredict_mask\ndice = {dice :.2f}")
axs[3].axis('off')

save_path = f'save_ans/completion_of_{save_path}'
plt.savefig(save_path)
plt.show()
