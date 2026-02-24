import io
import json
import os
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import SamModel, SamProcessor
import segmentation_models_pytorch as smp

from unetppSAM import run_inference_web

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "json", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[startup] Loading models on {DEVICE}...")

# Load models ONCE at startup
sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_weights_path = os.path.join(os.path.dirname(__file__), CONFIG["inference"]["load_state_dict"])
if os.path.exists(sam_weights_path):
    sam_model.load_state_dict(torch.load(sam_weights_path, map_location=DEVICE, weights_only=True))
    print(f"[startup] Loaded fine-tuned SAM weights from {sam_weights_path}")
else:
    print(f"[startup] {sam_weights_path} not found, using base SAM weights")
sam_model = sam_model.to(DEVICE).eval()
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

unet_model = smp.UnetPlusPlus(
    encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1
).to(DEVICE)
unet_weights_path = os.path.join(os.path.dirname(__file__), "unetplusplus_chkpt", "unetplusplus.pth")
if not os.path.exists(unet_weights_path):
    hf_repo = os.environ.get("HF_UNET_MODEL_REPO", "")
    if hf_repo:
        from huggingface_hub import hf_hub_download
        print(f"[startup] Downloading Unet++ weights from HF Hub: {hf_repo} ...")
        os.makedirs(os.path.dirname(unet_weights_path), exist_ok=True)
        unet_weights_path = hf_hub_download(repo_id=hf_repo, filename="unetplusplus.pth",
                                             local_dir=os.path.dirname(unet_weights_path))
if os.path.exists(unet_weights_path):
    unet_model.load_state_dict(torch.load(unet_weights_path, map_location=DEVICE, weights_only=True))
    print(f"[startup] Loaded Unet++ weights from {unet_weights_path}")
else:
    print(f"[startup] WARNING: {unet_weights_path} not found, Unet++ will produce random output")
unet_model.eval()
print("[startup] Models ready.")

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    bbox:  str        = Form(...),
    mask:  Optional[UploadFile] = File(default=None),
):
    bbox_list = [int(v) for v in json.loads(bbox)]
    pil_image = Image.open(io.BytesIO(await image.read()))
    pil_mask  = Image.open(io.BytesIO(await mask.read())) if mask else None
    result_png = run_inference_web(
        pil_image, bbox_list, sam_model, sam_processor, unet_model, DEVICE,
        gt_mask=pil_mask,
    )
    return Response(content=result_png, media_type="image/png")
