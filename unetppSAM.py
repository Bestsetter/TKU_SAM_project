import io
import json
import os
import random

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from transformers import SamModel, SamProcessor
from torchvision import transforms
import segmentation_models_pytorch as smp


def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    return [x_min, y_min, x_max, y_max]


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
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()
    if preds.device != targets.device:
        targets = targets.to(preds.device)
    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def _seg_metrics(pred, gt):
    """Compute Recall, Precision, Dice given two binary numpy arrays."""
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    recall    = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return recall, precision, dice


def run_inference_web(image, bbox, sam_model, sam_processor, unet_model, device, gt_mask=None):
    """
    Web inference entry point. No disk I/O.

    Parameters:
        image        : PIL Image (any mode, any size)
        bbox         : [x1, y1, x2, y2] in 256x256 coordinate space
        sam_model    : pre-loaded SamModel
        sam_processor: pre-loaded SamProcessor
        unet_model   : pre-loaded UnetPlusPlus
        device       : "cuda" or "cpu"
        gt_mask      : optional PIL Image — ground truth mask for metric computation

    Returns:
        PNG bytes.
        Without gt_mask: 1x3 figure [Original | SAM | Unet++]
        With    gt_mask: 1x4 figure [Original | GT  | SAM (metrics) | Unet++ (metrics)]
    """
    image_rgb  = image.convert("RGB").resize((256, 256))
    image_gray = image.convert("L").resize((256, 256))

    # SAM inference
    inputs = sam_processor(image_rgb, input_boxes=[[bbox]], return_tensors="pt").to(device)
    sam_model.eval()
    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)
    torch.cuda.empty_cache()
    medsam_seg = (torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze() > 0.5).astype(np.int32)

    # Unet++ inference
    preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image_tensor = preprocess(image_gray).unsqueeze(0).to(device)
    unet_model.eval()
    with torch.no_grad():
        pred_mask = unet_model(image_tensor)
        pred_mask = (torch.sigmoid(pred_mask) > 0.7).int()
    unet_seg = pred_mask[0, 0].cpu().numpy()

    img_arr = np.array(image_rgb)

    if gt_mask is not None:
        # Binarise GT
        gt = (np.array(gt_mask.convert("L").resize((256, 256))) > 0).astype(np.int32)

        sam_r,  sam_p,  sam_d  = _seg_metrics(medsam_seg, gt)
        unet_r, unet_p, unet_d = _seg_metrics(unet_seg,   gt)

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(img_arr);        axs[0].set_title("Original");      axs[0].axis("off")
        axs[1].imshow(img_arr);        show_mask(gt, axs[1], "blue")
        axs[1].set_title("Ground Truth"); axs[1].axis("off")
        axs[2].imshow(img_arr);        show_mask(medsam_seg, axs[2], "green")
        axs[2].set_title(f"SAM\nRecall={sam_r:.2f}  Prec={sam_p:.2f}\nDice={sam_d:.2f}")
        axs[2].axis("off")
        axs[3].imshow(img_arr);        show_mask(unet_seg, axs[3], "red")
        axs[3].set_title(f"Unet++\nRecall={unet_r:.2f}  Prec={unet_p:.2f}\nDice={unet_d:.2f}")
        axs[3].axis("off")
    else:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_arr);        axs[0].set_title("Original");         axs[0].axis("off")
        axs[1].imshow(img_arr);        show_mask(medsam_seg, axs[1], "green")
        axs[1].set_title("SAM Prediction");    axs[1].axis("off")
        axs[2].imshow(img_arr);        show_mask(unet_seg, axs[2], "red")
        axs[2].set_title("Unet++ Prediction"); axs[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close("all")
    buf.seek(0)
    return buf.read()


def _run_inference(config, image_path, mask_path, is_show_ans, is_gen_compare, is_show_compare):
    """Core inference logic: runs SAM (and optionally Unet++) on a given image/mask pair."""
    print(f"Image: {image_path}")
    print(f"Mask:  {mask_path}")

    image = Image.open(image_path).resize((256, 256))
    mask = Image.open(mask_path).resize((256, 256))

    ground_truth_mask = np.array(mask)
    prompt = get_bounding_box(ground_truth_mask)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- SAM inference ---
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    sam_model.load_state_dict(torch.load(config['load_state_dict'], map_location=device))
    sam_model = sam_model.to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
    sam_model.eval()
    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)
    torch.cuda.empty_cache()

    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.int32)

    # Save SAM result
    plt.imshow(medsam_seg)
    plt.axis("off")
    output_folder = config["ans_img_floder"]
    os.makedirs(output_folder, exist_ok=True)
    save_name = os.path.basename(image_path) + config["ans_img_name"]
    output_path = os.path.join(output_folder, save_name)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close("all")

    if is_show_ans:
        Image.open(output_path).show()

    if not is_gen_compare:
        return

    # --- Unet++ inference ---
    unet_model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(device)
    unet_model.load_state_dict(torch.load('unetplusplus_chkpt/unetplusplus.pth', map_location=device))

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image_tensor = preprocess(Image.open(image_path).convert('L')).unsqueeze(0).to(device)
    mask_tensor = preprocess(mask).unsqueeze(0).to(device)

    unet_model.eval()
    with torch.no_grad():
        pred_mask = unet_model(image_tensor)
        pred_mask = (torch.sigmoid(pred_mask) > 0.7).int()

    # --- Comparison plot ---
    _, axs = plt.subplots(1, 4)

    axs[0].imshow(image)
    axs[0].set_title("origin")
    axs[0].axis('off')

    axs[1].imshow(np.array(image))
    show_mask(np.array(mask), axs[1], "blue")
    axs[1].set_title("mask")
    axs[1].axis('off')

    medsam_seg_t = torch.tensor(medsam_seg)
    mask_tensor_sq = torch.squeeze(mask_tensor)
    sam_dice = dice_coefficient(medsam_seg_t, mask_tensor_sq)
    axs[2].imshow(np.array(image))
    show_mask(medsam_seg, axs[2], "green")
    axs[2].set_title(f"SAM's\npredict_mask\ndice = {sam_dice:.2f}")
    axs[2].axis('off')

    pred_mask_sq = torch.squeeze(pred_mask)
    unet_dice = dice_coefficient(pred_mask_sq, mask_tensor_sq).cpu().item()
    axs[3].imshow(np.array(image))
    show_mask(pred_mask[0, 0].cpu().numpy(), axs[3], "red")
    axs[3].set_title(f"Unet++'s\npredict_mask\ndice = {unet_dice:.2f}")
    axs[3].axis('off')

    output_folder = config["compare_all_floder"]
    os.makedirs(output_folder, exist_ok=True)
    save_name = os.path.basename(image_path) + config["compare_all_img_name"]
    output_path = os.path.join(output_folder, save_name)
    plt.savefig(output_path)
    plt.close("all")

    if is_show_compare:
        Image.open(output_path).show()


def gen_ans(config, is_show_ans=True, is_gen_compare=True, is_show_compare=True):
    """Randomly select an image from the test folders and run inference."""
    selected_folder = random.choice(config["test_folders"])
    image_files = [f for f in os.listdir(selected_folder) if f.endswith(config['test_img_path_endswith'])]
    selected_image = random.choice(image_files)
    image_path = os.path.join(selected_folder, selected_image)
    mask_path = image_path.replace(config['test_img_path_endswith'], config['test_img_mask_path_endswith'])
    _run_inference(config, image_path, mask_path, is_show_ans, is_gen_compare, is_show_compare)


def gen_ans_specific(config, is_show_ans=True, is_gen_compare=True, is_show_compare=True, image="sample.png"):
    """Run inference on a specific image file (mask expected at <image>_mask.png)."""
    image_path = image
    mask_path = image_path.replace(".png", "_mask.png")
    _run_inference(config, image_path, mask_path, is_show_ans, is_gen_compare, is_show_compare)


def gen_ans_directory(config, is_show_ans=True, is_gen_compare=True, is_show_compare=True, directory="sample"):
    """Run inference on all images (excluding masks) in a directory."""
    for filename in os.listdir(directory):
        if not filename.endswith(".png"):
            continue
        if filename.endswith("_mask.png"):
            continue
        file_path = os.path.join(directory, filename)
        gen_ans_specific(config, is_show_ans, is_gen_compare, is_show_compare, file_path)


if __name__ == "__main__":
    configfile_path = "json/config.json"
    with open(configfile_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    gen_ans(config["inference"])
