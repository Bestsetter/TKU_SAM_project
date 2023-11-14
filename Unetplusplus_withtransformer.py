import os
import pandas as pd
import glob
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

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
# print(dataset_df)

class BusiDataset(Dataset):
    """Pytorch dataset class for generating batch of transformed images
    Returns batch of images with different mask colors for each tumour
    class
    """
    def __init__(self, df: pd.DataFrame, input_size=(256, 256), transform=False):
        """
        Args:
            df (pd.DataFrame): _description_
            input_size (tuple, optional): _description_. Defaults to (256,256).
            transform (bool, optional): _description_. Defaults to False.
        """
        self.df = df
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # slice of the dataframe
        df_item = self.df.iloc[idx]

        # read the image and the mask
        img = cv.imread(df_item["images"], cv.IMREAD_GRAYSCALE)
        # Histogram equalization of input image
        # img = cv.equalizeHist(img)
        mask = cv.imread(df_item["masks"], cv.IMREAD_GRAYSCALE)
        # Convert image and maks to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        resize = transforms.Resize(size=self.input_size,antialias=True)
        img = resize(img)
        mask = resize(mask)

        return img, mask

def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice

import segmentation_models_pytorch as smp

device = 'cuda'if torch.cuda.is_available() else 'cpu'

model = smp.UnetPlusPlus(encoder_name="resnet34",
                        encoder_weights=None,
                        in_channels=1,
                        classes=1,
                        ).to(device)
                        
import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        lr: float = 3e-4,
        batch_size: int = 16,
        epochs: int = 100,
        device: str = "cuda:0",
    ) -> None:
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        # Global training variables
        self.BEST_VAL_LOSS = float("inf")
        self.BEST_EPOCH = 0

    def save_model(self, checkpoint_dir: str, checkpoint_name: str):
        date_postfix = datetime.now().strftime("%Y-%m-%d")
        model_name = f"{checkpoint_name}.pth"

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        else:
            print(f"[INFO:] Saving model to {os.path.join(checkpoint_dir,model_name)}")
            torch.save(
                self.model.state_dict(), os.path.join(checkpoint_dir, model_name)
            )
    def early_stopping(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        val_loss: float,
        epoch: int,
        patience: int = 10,
        min_delta: int = 0.01,
    ):
        if self.BEST_VAL_LOSS - val_loss >= min_delta:
            print(
                f"[INFO:] Validation loss improved from {self.BEST_VAL_LOSS} to {val_loss}"
            )
            print(f"[INFO] Current learning rate = {self.lr}")
            self.BEST_VAL_LOSS = val_loss
            self.BEST_EPOCH = epoch

            self.save_model(checkpoint_dir, checkpoint_name)
            return False
        if (
            self.BEST_VAL_LOSS - val_loss < min_delta
            and epoch - self.BEST_EPOCH >= patience
        ):
            return True
        return False
    
    @torch.no_grad()
    def evaluate(self,val_loader: DataLoader, desc="Validating") -> float:
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_. Defaults to "Validating".

        Returns:
            float: mean validation loss
        """
        progress_bar = tqdm(val_loader, total=len(val_loader))
        val_loss = []
        val_dice = []
        
        threshold = nn.Threshold(0.5,0.0)
        binary_ce_loss = nn.BCEWithLogitsLoss()

        for image, mask in progress_bar:
            image = image.to(self.device)
            mask = mask.to(self.device)
            # Get predictioin mask
            pred_mask = self.model(image)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = threshold(pred_mask)
            # get the dice loss
            dice = dice_coefficient(pred_mask, mask)

            dice_loss = 1 - dice
            ce_loss = binary_ce_loss(pred_mask, mask)
            loss = dice_loss + 0.2 * ce_loss
            val_loss.append(loss.detach().item())

            val_dice.append(dice.detach().item())
            # Update the progress bar
            progress_bar.set_description(desc)
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=np.mean(val_dice)
            )
            progress_bar.update()
        return np.mean(val_loss)
    
    @torch.no_grad()
    def test(self,test_loader: DataLoader, desc="Testing") -> float:
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_. Defaults to "Validating".

        Returns:
            float: mean validation loss
        """
        progress_bar = tqdm(test_loader, total=len(test_loader))
        val_loss = []
        dice_scores = []
        
        threshold = nn.Threshold(0.5,0.0)
        for image, mask in progress_bar:
            image = image.to(self.device)
            mask = mask.to(self.device)
            # Get predictioin mask
            pred_mask = self.model(image)
            
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = threshold(pred_mask)
            
            # get the dice loss
            dice = dice_coefficient(pred_mask, mask)

            dice_loss = 1 - dice

            loss = dice_loss

            val_loss.append(loss.item())

            dice_scores.append(dice.data.detach().item())
            # Update the progress bar
            progress_bar.set_description(desc)
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=1 - np.mean(val_loss)
            )
            progress_bar.update()
        return np.mean(val_loss), dice_scores
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str,
        checkpoint_name: str,
    ):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.01, verbose=True
        )
        threshold = nn.Threshold(0.5,0.)
        binary_ce_loss = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = []
            epoch_dice = []
            progress_bar = tqdm(train_loader, total=len(train_loader))
            for image, mask in progress_bar:
                image = image.to(self.device)
                mask = mask.to(self.device)
                # Get predictioin mask
                pred_mask = self.model(image)

                pred_mask = torch.sigmoid(pred_mask)
                
                # Calculate loss
                dice = dice_coefficient(pred_mask, mask)

                dice_loss = 1 - dice

                ce_loss = binary_ce_loss(pred_mask, mask)
                loss = dice_loss + 0.2 * ce_loss

                epoch_loss.append(loss.detach().item())

                epoch_dice.append(dice.detach().item())
                # empty gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.set_description(f"Epoch {epoch+1}/{self.epochs}")
                progress_bar.set_postfix(
                    loss=np.mean(epoch_loss), dice=np.mean(epoch_dice)
                )
                progress_bar.update()

            if epoch % 2 == 0:
                validation_loss = self.evaluate(val_loader, desc=f"Eval"
                )
                scheduler.step(torch.tensor(validation_loss))


                if self.early_stopping(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                    val_loss=validation_loss,
                    epoch=epoch,
                ):
                    print(f"[INFO:] Early Stopping!!")
                    break

from sklearn.model_selection import KFold, train_test_split

train_df, test_df = train_test_split(dataset_df, train_size=0.8, random_state=42)
input_shape = (256,256)

train_ds = BusiDataset(train_df, input_size=input_shape)
test_ds = BusiDataset(test_df, input_size=input_shape)
batch_size = 15

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
trainer = Trainer(model,lr=3e-4,batch_size=batch_size)
trainer.train(train_loader,test_loader,"unetplusplus_chkpt","unetplusplus")
loss,dice_scores = trainer.test(test_loader)
print(f"Mean Dice = {np.mean(dice_scores)}")
image_iter = iter(test_loader)