---
title: TKU SAM Breast Tumor Segmentation
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TKU SAM Project — 乳房腫瘤切割 Web Demo

使用 BUSI（Breast Ultrasound Images）資料集，結合 SAM 與 Unet++ 兩種模型進行乳房腫瘤切割，並以 Web 介面展示結果。

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/yuchengChang/tku-sam)

> **線上 Demo**：[https://huggingface.co/spaces/yuchengChang/tku-sam](https://huggingface.co/spaces/yuchengChang/tku-sam)
> 無需安裝，直接上傳超音波圖片即可使用。

---

## 架構

| 模型 | 說明 |
|------|------|
| **SAM** (facebook/sam-vit-base) | 使用者畫出 bounding box 後進行切割，可選擇性載入 fine-tuned 權重 |
| **Unet++** (ResNet34 encoder) | 以灰階超音波影像訓練的切割模型，輸出與 SAM 並列比較 |

---

## 資料集

[BUSI Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)（需手動下載）

放置於：
```
Dataset_BUSI_with_GT/
├── benign/       # 437 張
├── malignant/    # 210 張
└── normal/
```

---

## 安裝

```bash
pip install -r requirements.txt
```

---

## 訓練

### Unet++
```bash
python train_unet.py
```
輸出：`unetplusplus_chkpt/unetplusplus.pth`（約 100MB，需 10-20 分鐘）

### SAM Fine-tune（選用）
不 fine-tune 也可使用，但精度較低。
```bash
pip install monai
python train_sam.py
```
輸出：`best.pth`（約 30-60 分鐘）

---

## 啟動 Web Demo

```bash
# 啟動後端
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# 對外公開（需安裝 ngrok）
ngrok http 8000
```

開啟 `http://localhost:8000/` 即可使用。

---

## 使用方式

1. 上傳 BUSI 超音波圖片（PNG / JPG）
2. （選用）上傳對應的 Ground Truth Mask，可額外顯示 Recall / Precision / Dice 指標
3. 在腫瘤位置拖曳滑鼠畫出黃色 bounding box
4. 點擊 **Predict**
5. 頁面顯示對比圖：
   - 未上傳 mask：3 格（原圖 ｜ SAM 切割綠色 ｜ Unet++ 切割紅色）
   - 已上傳 mask：4 格（原圖 ｜ GT 藍色 ｜ SAM + 指標 ｜ Unet++ + 指標）

---

## 專案結構

```
TKU_SAM_project/
├── app.py                  # FastAPI 後端
├── unetppSAM.py            # 推論核心（含 run_inference_web）
├── train_unet.py           # Unet++ 訓練腳本
├── train_sam.py            # SAM fine-tune 腳本
├── templates/
│   └── index.html          # 前端單頁介面
├── json/
│   └── config.json         # 模型路徑設定
├── Dataset_BUSI_with_GT/   # 資料集（不含在 git）
├── unetplusplus_chkpt/
│   └── unetplusplus.pth    # Unet++ 權重（不含在 git）
├── best.pth                # SAM fine-tuned 權重（不含在 git，選用）
└── requirements.txt
```

---

## 訓練結果

| 模型 | Val Dice | Val Loss |
|------|----------|----------|
| Unet++ (30 epochs) | **0.707** | 0.369 |
| SAM (base, no fine-tune) | — | — |
