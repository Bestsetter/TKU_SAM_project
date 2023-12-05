import torch
import torch.nn as nn

class resNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1,img_width=256):# input img 256X256
        super(resNet, self).__init__() 
        self.out_channel = out_channel
        self.x7conv3to64 = nn.Conv2d(in_channel,64,7,1,3,bias=False)
        self.x1conv64to64 = nn.Conv2d(64,64,1,1)
        self.x1conv128to64 = nn.Conv2d(128,64,1,1)


        self.x3conv64to64 = nn.Conv2d(64,64,3,1,1,bias=False)
        self.x3conv64to128 = nn.Conv2d(64,128,3,1,1,bias=False)
        self.x3conv128to64 = nn.Conv2d(128,64,3,1,1,bias=False)
        self.x3conv128to128 = nn.Conv2d(128,128,3,1,1,bias=False)
        self.x3conv128to256 = nn.Conv2d(128,256,3,1,1,bias=False)
        self.x3conv256to128 = nn.Conv2d(256,128,3,1,1,bias=False)
        self.x3conv256to256 = nn.Conv2d(256,256,3,1,1,bias=False) 
        self.x3conv256to512 = nn.Conv2d(256,512,3,1,1,bias=False) 
        self.x3conv512to256 = nn.Conv2d(512,256,3,1,1,bias=False)
        self.x3conv512to512 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.x3conv512to1024 = nn.Conv2d(512,1024,3,1,1,bias=False)
        self.x3conv1024to512 = nn.Conv2d(1024,512,3,1,1,bias=False)
        self.x3conv1024to1024 = nn.Conv2d(1024,1024,3,1,1,bias=False)

        self.normal64 = nn.BatchNorm2d(64)
        self.normal128 = nn.BatchNorm2d(128)
        self.normal256 = nn.BatchNorm2d(256)
        self.normal512 = nn.BatchNorm2d(512)
        self.normal1024 = nn.BatchNorm2d(1024)
        self.d1normal1024 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2)
        self.fc_pool = nn.AvgPool2d(int(img_width/16), stride=1)

        self.linear1024to1024 = nn.Linear(1024, 1024)
        self.linear1024to2 = nn.Linear(1024, 3)
        self.drop = nn.Dropout(0.1)

        # torch.cat([x, y], dim=1)
        

    
    def forward(self, x):
        output = self.x7conv3to64(x)
        output = self.pool(output)
        # print('1',output.shape)
        res = self.x3conv64to64(output)
        res = self.x1conv64to64(res)
        # print('2',output.shape, res.shape)
        output = self.normal64(output)
        output = self.relu(self.x3conv64to64(output))
        output = self.relu(self.x3conv64to64(output))
        
        output = torch.cat([output, res], dim=1)
        res = self.x1conv128to64(output)
        
        output = self.relu(self.x3conv128to64(output))
        output = self.normal64(output)
        output = self.relu(self.x3conv64to64(output))

        output = torch.cat([output, res], dim=1)
        res = self.x1conv128to64(output)
        
        output = self.relu(self.x3conv128to64(output))
        output = self.normal64(output)
        output = self.relu(self.x3conv64to64(output))

        output = torch.cat([output, res], dim=1)  # 1
        output = self.x3conv128to128(output)
        output = self.pool(output)
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1) # 2
        output = self.x3conv256to128(output)
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 3
        output = self.x3conv256to128(output)
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 4
        output = self.x3conv256to128(output)
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 1
        output = self.x3conv256to256(output)
        output = self.pool(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))
        
        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to512(output)
        res = output
        output = self.normal512(output)
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to512(output)
        res = output
        output = self.normal512(output)
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to512(output)
        res = output
        output = self.normal512(output)
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to1024(output)
        output = self.pool(output)
        # print('4.4', output.shape)
        output = self.fc_pool(output)
        # print('4.5', output.shape)
        output = output.reshape(output.shape[0], -1)
        output = self.d1normal1024(output)
        # print('5', output.shape)
        output = self.drop(self.linear1024to1024(output))
        output = self.d1normal1024(output)
        output = self.relu(self.linear1024to1024(output))
        output = self.drop(self.linear1024to1024(output))
        output = self.d1normal1024(output)
        output = self.relu(self.linear1024to2(output))
        # print('6', output.shape)
        return output
        
import cv2 as cv 
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


class data_classif(Dataset):
    def __init__(self, image_pd, input_size=(256, 256)):
        self.image_pd = image_pd
        self.input_size = input_size
    def __len__(self):
        return len(self.image_pd)
    def __getitem__(self, index):
        df_item = self.image_pd.iloc[index]
        image = cv.imread(df_item['images'], cv.IMREAD_GRAYSCALE)
        
        if "benign" in df_item['images']:
            target = torch.tensor([1,0,0])
        elif "malignant" in df_item['images']:
            target = torch.tensor([0,1,0])
        elif "normal" in df_item['images']:
            target = torch.tensor([0,0,1])
        else:
            print("data error")

        image = TF.to_tensor(image)

        resize = transforms.Resize(size=self.input_size,antialias=True)
        image = resize(image)
        return image, target

def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice

def CNN_train(
    epoch=3,
    batch=5
):
    import os
    import pandas as pd
    import glob
    busi_dataset_path = "Dataset_BUSI_with_GT"
    import re
    """ Benign """
    benign_path = os.path.join(busi_dataset_path,"benign")
    benign_images = sorted(glob.glob(benign_path +"/*).png"))
    benign_masks = sorted(glob.glob(benign_path +"/*mask.png"))
    key = [int(re.findall(r'[0-9]+',image_name)[0]) for image_name in benign_images]
    benign_df = pd.DataFrame({'key':key,'images':benign_images,'masks':benign_masks})

    """ Malignant"""
    malignant_path = os.path.join(busi_dataset_path,"malignant")
    malignant_images = sorted(glob.glob(malignant_path +"/*).png"))
    malignant_masks = sorted(glob.glob(malignant_path +"/*mask.png"))
    key = [int(re.findall(r'[0-9]+',image_name)[0]) + 437 for image_name in malignant_images]
    malignant_df = pd.DataFrame({'key':key,'images':malignant_images,'masks':malignant_masks})

    """ Normal """

    normal_path = os.path.join(busi_dataset_path,"normal")
    normal_images = sorted(glob.glob(malignant_path +"/*).png"))
    normal_masks = sorted(glob.glob(malignant_path +"/*mask.png"))
    key = [int(re.findall(r'[0-9]+',image_name)[0]) + 648 for image_name in normal_images]
    normal_df = pd.DataFrame({'key':key,'images':normal_images,'masks':normal_masks})
    dataset_df = pd.concat([benign_df,malignant_df,normal_df])

    device = 'cuda'if torch.cuda.is_available() else 'cpu'
    model = resNet(1,1).to(device=device)
    data = data_classif(dataset_df)

    from torch.utils.data import random_split
    train_data, test_data = random_split(data, [int(len(data)*0.9), len(data)-int(len(data)*0.9)])
    
    from torch.utils.data import DataLoader
    train_deter_Loader = DataLoader(train_data, batch, shuffle=True, drop_last=True)
    test_deter_Loader = DataLoader(test_data, batch, shuffle=False, drop_last=True)

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), 0.001)
    loss_f = nn.CrossEntropyLoss() if model.out_channel > 1 else nn.BCEWithLogitsLoss()

    for e in range(1, epoch):
        epoch_loss = 0
        # from tqdm import tqdm
        for idx, (predictor, target) in enumerate(train_deter_Loader):
            predictor = predictor.to(device, torch.float32)
            target = target.to(device, torch.float32)
            prediction = model(predictor)
            loss = loss_f(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"\t\tBatch {idx+1} done, with loss = {loss}")
        print(f"[+] epoch {epoch+1} done, with loss = {epoch_loss/len(train_deter_Loader)}")
    return model


model = CNN_train()



