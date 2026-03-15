import base64
import io
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import SamModel, SamProcessor
import segmentation_models_pytorch as smp

from unetppSAM import run_inference_web
from cnn_classifier import ResNetClassifier

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "json", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model references — set by background loader
_sam_model = None
_sam_processor = None
_unet_model = None
_cnn_model = None
_models_ready = False
_load_error: Optional[str] = None


def _load_models():
    global _sam_model, _sam_processor, _unet_model, _cnn_model, _models_ready, _load_error
    try:
        print(f"[loader] Loading models on {DEVICE}...")

        # --- SAM ---
        sam = SamModel.from_pretrained("facebook/sam-vit-base")
        sam_weights_path = os.path.join(os.path.dirname(__file__), CONFIG["inference"]["load_state_dict"])
        if not os.path.exists(sam_weights_path):
            hf_repo = os.environ.get("HF_UNET_MODEL_REPO", "")
            if hf_repo:
                from huggingface_hub import hf_hub_download
                print(f"[loader] Downloading SAM weights from {hf_repo} ...")
                sam_weights_path = hf_hub_download(
                    repo_id=hf_repo, filename="best.pth",
                    local_dir=os.path.dirname(os.path.join(os.path.dirname(__file__), CONFIG["inference"]["load_state_dict"])),
                )
        if os.path.exists(sam_weights_path):
            sam.load_state_dict(torch.load(sam_weights_path, map_location=DEVICE, weights_only=True))
            print(f"[loader] Loaded fine-tuned SAM weights from {sam_weights_path}")
        else:
            print("[loader] best.pth not found — using base SAM weights")
        _sam_model = sam.to(DEVICE).eval()
        _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        # --- Unet++ ---
        unet = smp.UnetPlusPlus(
            encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1
        ).to(DEVICE)
        unet_weights_path = os.path.join(os.path.dirname(__file__), "unetplusplus_chkpt", "unetplusplus.pth")
        if not os.path.exists(unet_weights_path):
            hf_repo = os.environ.get("HF_UNET_MODEL_REPO", "")
            if hf_repo:
                from huggingface_hub import hf_hub_download
                print(f"[loader] Downloading Unet++ weights from {hf_repo} ...")
                os.makedirs(os.path.dirname(unet_weights_path), exist_ok=True)
                unet_weights_path = hf_hub_download(
                    repo_id=hf_repo, filename="unetplusplus.pth",
                    local_dir=os.path.dirname(unet_weights_path),
                )
        if os.path.exists(unet_weights_path):
            unet.load_state_dict(torch.load(unet_weights_path, map_location=DEVICE, weights_only=True))
            print(f"[loader] Loaded Unet++ weights from {unet_weights_path}")
        else:
            print("[loader] WARNING: unetplusplus.pth not found — Unet++ will produce random output")
        _unet_model = unet.eval()

        # --- CNN ---
        cnn = ResNetClassifier(in_channel=1, num_classes=2).to(DEVICE)
        cnn_weights_path = os.path.join(os.path.dirname(__file__), "cnn_chkpt", "cnn_best.pth")
        if not os.path.exists(cnn_weights_path):
            hf_repo = os.environ.get("HF_UNET_MODEL_REPO", "")
            if hf_repo:
                from huggingface_hub import hf_hub_download
                print(f"[loader] Downloading CNN weights from {hf_repo} ...")
                os.makedirs(os.path.dirname(cnn_weights_path), exist_ok=True)
                cnn_weights_path = hf_hub_download(
                    repo_id=hf_repo, filename="cnn_best.pth",
                    local_dir=os.path.dirname(cnn_weights_path),
                )
        if os.path.exists(cnn_weights_path):
            cnn.load_state_dict(torch.load(cnn_weights_path, map_location=DEVICE, weights_only=True))
            print(f"[loader] Loaded CNN weights from {cnn_weights_path}")
        else:
            print("[loader] WARNING: cnn_best.pth not found — CNN will produce random output")
        _cnn_model = cnn.eval()

        _models_ready = True
        print("[loader] All models ready.")
    except Exception as e:
        _load_error = str(e)
        print(f"[loader] FATAL: model loading failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in background thread so HTTP server starts immediately
    t = threading.Thread(target=_load_models, daemon=True)
    t.start()
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/health")
async def health():
    if _load_error:
        return JSONResponse({"status": "error", "detail": _load_error}, status_code=500)
    if not _models_ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    bbox:  str        = Form(...),
    mask:  Optional[UploadFile] = File(default=None),
):
    if not _models_ready:
        msg = _load_error if _load_error else "Models are still loading, please retry in a moment."
        return JSONResponse({"error": msg}, status_code=503)

    bbox_list = [int(v) for v in json.loads(bbox)]
    pil_image = Image.open(io.BytesIO(await image.read()))
    pil_mask  = Image.open(io.BytesIO(await mask.read())) if mask else None
    result_png, label, prob_benign, prob_malignant = run_inference_web(
        pil_image, bbox_list, _sam_model, _sam_processor, _unet_model, DEVICE,
        gt_mask=pil_mask, cnn_model=_cnn_model,
    )
    return JSONResponse({
        "image": base64.b64encode(result_png).decode(),
        "label": label,
        "prob_benign":    round(prob_benign,    4),
        "prob_malignant": round(prob_malignant, 4),
    })
