import numpy as np
import os
from PIL import Image

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import SamProcessor
import torch
from transformers import SamModel
import random
from datasets import DatasetDict, load_dataset
import os

from torchvision import transforms
import segmentation_models_pytorch as smp

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

def show_mask(mask, ax, color=False, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color == "red":
        color = np.array([255/255, 70/255, 70/255, 0.5])
    elif color == "green":
        color = np.array([70/255, 255/255, 70/255, 0.5])
    elif color == "blue":
        color = np.array([70/255, 150/255, 255/255, 0.5])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    # print("mask.shape = ", mask.shape)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    if preds.device != targets.device:
        # 將targets移動到與preds相同的設備上
        targets = targets.to(preds.device)

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice


def gen_ans(config, is_show_ans = True, is_gen_compare = True, is_show_compare = True):
    print("Generating ans")
    #載入model 架構
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.load_state_dict(torch.load(config['load_state_dict']))
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    #設定測試圖片資料夾(隨機選取圖片，normal 不適用)
    
    selected_folder = random.choice(config["test_folders"])
    image_files = [f for f in os.listdir(selected_folder) if f.endswith(config['test_img_path_endswith'])]
    selected_image = random.choice(image_files)
    image_path = os.path.join(selected_folder, selected_image)
    mask_path = image_path.replace(config['test_img_path_endswith'], config['test_img_mask_path_endswith'])
    print(image_path)
    print(mask_path)
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    image = image.resize((256, 256))
    mask = mask.resize((256, 256))
    # print(image.size)
    # print(mask.size)

    ground_truth_mask = np.array(mask)
    prompt = get_bounding_box(ground_truth_mask)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # prepare image + box prompt for the model
    inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
    # for k,v in inputs.items():
    #   print(k,v.shape)

    model.eval()

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    torch.cuda.empty_cache()
    
    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.int32)
    
    plt.imshow(medsam_seg)
    plt.axis("off")
    
    output_folder = config["ans_img_floder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    save_name = os.path.basename(image_path)
    save_name += config["ans_img_name"]
    output_path = os.path.join(output_folder, save_name)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

    if is_show_ans:
        img = Image.open(output_path)
        img.show()

    if is_gen_compare:
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(device)
        model.load_state_dict(torch.load('unetplusplus_chkpt/unetplusplus.pth'))

        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        uimage = Image.open(image_path).convert('L')

        # print("mask\n", mask)
        image_tensor = preprocess(uimage).unsqueeze(0).to(device)
        mask_tensor = preprocess(mask).unsqueeze(0).to(device)
        # print("mask_tensor\n", mask_tensor)

        with torch.no_grad():
            model.eval()
            pred_mask = model(image_tensor)
            pred_mask = (torch.sigmoid(pred_mask) > 0.7).int()
            # dice = dice_coefficient(pred_mask,mask_tensor).cpu().item()

        # print("Dice = ", dice)

        _, axs = plt.subplots(1, 4)

        axs[0].imshow(image)
        axs[0].set_title("origin")
        axs[0].axis('off')

        axs[1].imshow(np.array(image))
        ground_truth_seg = np.array(mask)
        show_mask(ground_truth_seg, axs[1], "blue")
        axs[1].title.set_text(f"mask")
        axs[1].axis('off')

        axs[2].imshow(np.array(image))
        show_mask(medsam_seg, axs[2], "green")
        axs[2].set_title('SAM predict')
        medsam_seg = torch.tensor(medsam_seg)
        mask_tensor = torch.squeeze(mask_tensor)
        dice = dice_coefficient(medsam_seg, mask_tensor)
        axs[2].set_title(f"SAM's\npredict_mask\ndice = {dice :.2f}")
        axs[2].axis('off')


        axs[3].imshow(np.array(image))
        show_mask(pred_mask[0, 0].cpu().numpy(), axs[3], "red")
        pred_mask = torch.squeeze(pred_mask)
        dice = dice_coefficient(pred_mask, mask_tensor).cpu().item()
        axs[3].set_title(f"Unet++'s\npredict_mask\ndice = {dice :.2f}")
        axs[3].axis('off')

        # 疊圖
        # axs[4].imshow(np.array(image), alpha=0.3)
        # ground_truth_seg = np.array(mask)
        # show_mask(ground_truth_seg, axs[4], "blue")
        # show_mask(medsam_seg, axs[4], "green")
        # show_mask(pred_mask, axs[4], "red")
        # axs[4].title.set_text(f"All in one")
        # axs[4].axis('off')

        output_folder = config["compare_all_floder"]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        save_name = os.path.basename(image_path)
        save_name += config["compare_all_img_name"]
        output_path = os.path.join(output_folder, save_name)
        plt.savefig(output_path)
        if is_show_compare:
            img = Image.open(output_path)
            img.show()
        plt.close("all")

    
if __name__ == "__main__":
    import json
    configfile_path = "json/config.json"
    configfile = open(configfile_path, "r",encoding="utf-8").read()
    config = json.loads(configfile)
    gen_ans(config["oentheSAM.py"])