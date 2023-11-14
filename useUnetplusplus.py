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

model = smp.UnetPlusPlus(encoder_name="resnet34",
                        encoder_weights=None,
                        in_channels=1,
                        classes=1,
                        ).to(device)

model.load_state_dict(torch.load('unetplusplus_chkpt/unetplusplus.pth'))

dataset_path = 'Dataset_BUSI_with_GT/benign/'
image_path = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(').png')]
image_path = random.choice(image_path)
print(image_path)
mask_path = image_path.replace('.png', '_mask.png')
print(mask_path)

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice

image = Image.open(image_path).convert('L')
mask = Image.open(mask_path)
image = image.resize((256, 256))
mask = mask.resize((256, 256))

image_tensor = preprocess(image).unsqueeze(0).to(device)
mask_tensor = preprocess(mask).unsqueeze(0).to(device)

with torch.no_grad():
    model.eval()
    pred_mask = model(image_tensor)
    pred_mask = (torch.sigmoid(pred_mask) > 0.7).int()
    dice = dice_coefficient(pred_mask,mask_tensor).cpu().item()

fig, axs = plt.subplots(1,3,figsize=(25,5), gridspec_kw={'wspace': 0.3, 'hspace': 0, 'width_ratios': [1, 1, 1]})
axs[0].imshow(image)
axs[0].set_title('Image')
axs[0].axis('off')
axs[1].imshow(mask)
axs[1].set_title('Mask')
axs[1].axis('off')
axs[2].imshow(pred_mask[0, 0].cpu().numpy(), cmap='gray')  # 取第一張圖的第一個通道的內容
axs[2].set_title('Predicted Mask')
axs[2].set_title(f"predict_mask\ndice = {dice :.2f}")
axs[2].axis('off')

plt.show()
