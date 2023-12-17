import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

def import_images(folder,target):
    images = []
    for item in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,item),0)
        if img is not None:
            images.append([img,target])
    return images

# 類別
print(os.listdir("D:/TKU_SAM_project/Dataset_BUSI_with_GT/"))
# ['benign', 'malignant', 'normal']

bengin = import_images("D:/TKU_SAM_project/Dataset_BUSI_with_GT/benign/",0)
malignant = import_images("D:/TKU_SAM_project/Dataset_BUSI_with_GT/malignant/",1)
normal = import_images("D:/TKU_SAM_project/Dataset_BUSI_with_GT/normal/",2)

full_data = bengin+malignant+normal

feature_matrix = []
label = []
for x,y in full_data:
    feature_matrix.append(x)
    label.append(y)

## resized image
X=[]
img_size=256

for x in feature_matrix:
    new_array = cv2.resize(x,(img_size,img_size))
    new_array = new_array/255   #[0, 1]
    X.append(new_array)

print(np.array(X).shape)
print(np.array(X).shape[0])
print(np.array(X).shape[1])
print(np.array(X).shape[2])

X_M = np.array(X)
X_M_R = X_M.reshape(X_M.shape[0], X_M.shape[1], X_M.shape[2], 1)
X_train,X_test,y_train,y_test = train_test_split(X_M_R,label)

print("X_train size: ",X_train.shape)
print("X_test Size: ",X_test.shape)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, do_batch_norm=True):
        super(Conv2dBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if do_batch_norm else None
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if do_batch_norm else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) if self.bn1 else F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x))) if self.bn2 else F.relu(self.conv2(x))
        return x

class Resnet(nn.Module):
    def __init__(self, num_filters=16, dropout=0.1, do_batch_norm=True):
        super(Resnet, self).__init__()

        self.conv1 = Conv2dBlock(1, num_filters * 1, do_batch_norm=do_batch_norm)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = Conv2dBlock(num_filters * 1, num_filters * 2, do_batch_norm=do_batch_norm)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = Conv2dBlock(num_filters * 2, num_filters * 4, do_batch_norm=do_batch_norm)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(dropout)

        self.conv4 = Conv2dBlock(num_filters * 4, num_filters * 8, do_batch_norm=do_batch_norm)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(dropout)

        self.conv5 = Conv2dBlock(num_filters * 8, num_filters * 16, do_batch_norm=do_batch_norm)

        self.up6 = nn.ConvTranspose2d(num_filters * 16, num_filters * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = Conv2dBlock(num_filters * 16, num_filters * 8, do_batch_norm=do_batch_norm)

        self.up7 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = Conv2dBlock(num_filters * 8, num_filters * 4, do_batch_norm=do_batch_norm)

        self.up8 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = Conv2dBlock(num_filters * 4, num_filters * 2, do_batch_norm=do_batch_norm)

        self.up9 = nn.ConvTranspose2d(num_filters * 2, num_filters * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = Conv2dBlock(num_filters * 2, num_filters * 1, do_batch_norm=do_batch_norm)

        self.conv10 = nn.Conv2d(num_filters * 1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.drop1(self.pool1(c1))

        c2 = self.conv2(p1)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.conv3(p2)
        p3 = self.drop3(self.pool3(c3))

        c4 = self.conv4(p3)
        p4 = self.drop4(self.pool4(c4))

        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        output = F.sigmoid(self.conv10(c9))
        return output

X_train_tensor = torch.Tensor(X_train).unsqueeze(1).squeeze(4).to("cuda:0")
y_train_tensor = torch.Tensor(y_train).unsqueeze(1)

model = Resnet(num_filters=16, dropout=0.1, do_batch_norm=True)
model = model.to("cuda:0")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_outputs = (test_outputs > 0.5).float()

accuracy = (test_outputs == y_test).float().mean()
print(f'Test Accuracy: {accuracy.item()}')
sample_index = 0  # 选择一个样本
sample_input = X_test[sample_index].unsqueeze(0)
sample_output = model(sample_input)
sample_prediction = (sample_output > 0.5).float()

# 将 PyTorch 张量转换为 NumPy 数组以便显示
sample_input_np = sample_input.squeeze().numpy()
sample_output_np = sample_output.squeeze().numpy()
sample_prediction_np = sample_prediction.squeeze().numpy()

# 可视化样本
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(sample_input_np, cmap='gray')
plt.title('Input Image')

plt.subplot(132)
plt.imshow(sample_output_np, cmap='gray')
plt.title('Model Output')

plt.subplot(133)
plt.imshow(sample_prediction_np, cmap='gray')
plt.title('Binary Prediction')

plt.show()