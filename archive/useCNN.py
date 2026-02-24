import torch
import torch.nn as nn
import cv2 as cv
from torchvision import transforms
import torchvision.transforms.functional as TF
from wall_cnn import CNN_train

# model = resNet(1,2).to('cuda')
# model.load_state_dict(torch.load("./cnn_model/e3_acc93.75%.pth"))
model = CNN_train()
# 自己設要測試的資料
# image = cv.imread("Dataset_BUSI_with_GT/malignant/malignant (1).png", cv.IMREAD_GRAYSCALE)
# image = TF.to_tensor(image)
# resize = transforms.Resize(size=(256,256),antialias=True)
# image = resize(image).to('cuda').unsqueeze(0)
# print("img", image.shape)
# print("[benign, malignant, normal]")
# prediction = torch.sigmoid(model(image))
# prediction = torch.where(prediction > 0.5, 1, 0)
# print(prediction)