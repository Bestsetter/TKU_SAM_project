import torch
import torch.nn as nn
import glob
import os
import re
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Unetplusplus_withtransformer import BusiDataset, Trainer


class ULayerDown(nn.Module):
    """Downward layer of UNET module, contains a 2D convolution layer, an activation function and a Maxpooling layer.
Stores its forward outputs for referencing by upwards layers."""
    def __init__(self,channels_in,channels_out, **layer_kwargs):

        super().__init__()
        self.conv_layer = nn.Conv2d(channels_in,channels_out,3,**layer_kwargs)
        self.activation = nn.ReLU()
        self.scale_layer = nn.MaxPool2d(2,padding=1)


        self.output = None

    def forward(self,X) :
            Y = self.conv_layer(X)
            Y = self.activation(Y)
            Y = nn.Dropout(0.4)(Y)
            Y = self.scale_layer(Y)
            self.output = Y
            return Y


class ULayerUp(nn.Module) :
    def __init__(self,single_feature_map_channels,n_classes,mask_shape,connected_nodes,_device="cuda", **layer_kwargs):
        """Upwards layer of UNET. handles concatenation and down/upsampling according to source layers and their levels.
        produces a side output corresponding to the mask predicted at this scale"""

        super().__init__()


        self.channels_per_fmap = single_feature_map_channels
        self.conv_layer = nn.Conv2d(self.channels_per_fmap*(len(connected_nodes)+1), self.channels_per_fmap, 3,padding=1, **layer_kwargs)
        self.normed = nn.BatchNorm2d(self.channels_per_fmap)
        self.activation = nn.ReLU()
        self.scale_layer = nn.Upsample(scale_factor=2)
        self.output = None

        self.device = _device

        self.prep_layers = []
        self.sources = connected_nodes

        self.side_mask_predictor = nn.Sequential(nn.Conv2d(self.channels_per_fmap,n_classes,3),
                                                  nn.Sigmoid(),
                                                  nn.Upsample(mask_shape)
                                                  )

        self.side_mask_output = None

        # add appropriate scaling/conv pipeline to the output of other layers

        for node in connected_nodes :
            self.prep_layers.append(nn.Conv2d(node.conv_layer.out_channels,
                                                              self.channels_per_fmap,3,padding=1).to(self.device))

    def forward(self,X) :

        to_cat = [X]
        for i in range(len(self.sources )) :
            resized_output = nn.Upsample(X.shape[-2:],mode="nearest")(self.sources[i].output)
            Xc = self.prep_layers[i](resized_output)
            to_cat.append(Xc)
        in_tensor = torch.cat(to_cat,1)
        Y = self.conv_layer(in_tensor)
        Y = self.normed(Y)
        Y = self.activation(Y)
        Y = nn.Dropout(0.4)(Y)

        self.output = Y
        if  self.training and self.scale_layer is not None and self.side_mask_predictor is not None:
            Y = self.scale_layer(Y)
            self.side_mask_output = self.side_mask_predictor(Y)
        return Y

class ClassificationArm(nn.Module) :
    """proposed classification module to make the model more robust to negative images by predicting class presence
    in images from the deepest encoding layer and multiplying output masks during training.

    """
    def __init__(self,in_channels,n_classes,input_size=25):
        super().__init__()

        pooled_size = input_size//2

        self.sequence = nn.Sequential(nn.Upsample([input_size,input_size]),
                                       nn.Conv2d(in_channels,n_classes,3,padding=1),
                                       nn.LeakyReLU(),
                                       nn.AdaptiveMaxPool2d(pooled_size),
                                       nn.Flatten(start_dim=1),
                                       nn.Linear(n_classes*pooled_size*pooled_size,96),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.4),
                                       nn.Linear(96,64),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.4),
                                       nn.Linear(64,n_classes))
        self.sig_layer = nn.Sigmoid()
    def forward(self,X):
        Y = self.sequence(X)
        return self.sig_layer(Y)





class UNet3Plus(nn.Module) :
    """UNET3+ autoencoder for semantic segmentation."""
    def __init__(self,
                 n_classes = 1,
                 in_channels = 1,
                 depth = 5,
                 first_output_channels = 16,
                 upwards_feature_channels = 64,
                 sideways_mask_shape = [256,256]):

        super().__init__()

        input_layer = ULayerDown(in_channels,first_output_channels)

        down_layers = [ULayerDown(first_output_channels*2**i,first_output_channels*2**(i+1)) for i in range(depth-1)]
        down_layers.append(ULayerDown(first_output_channels*2**(depth-1),upwards_feature_channels))
        down_layers.insert(0,input_layer)

        self.seqdown = nn.Sequential(*down_layers)

        up_layers = []

        for i in range(depth) :

            up_layers.append(ULayerUp(upwards_feature_channels,
                            n_classes,
                            mask_shape = sideways_mask_shape,
                            connected_nodes= down_layers[:depth-1-i]+ up_layers ))

        self.sequp = nn.Sequential(*up_layers)

        self.classifier = ClassificationArm(upwards_feature_channels,n_classes)

        self.last_layer = nn.Conv2d(upwards_feature_channels,n_classes,3,padding=1)

        self._side_mask_shape = sideways_mask_shape

        self.presence_prediction = None

    def forward(self,X):

        Y = nn.functional.normalize(X,dim=1)
        Y = self.seqdown(Y)
        Y = nn.Upsample(scale_factor=2)(Y)
        self.presence_prediction = self.classifier(Y)

        Y = self.sequp(Y)

        for layer in self.sequp :
            layer.side_mask_output *= self.presence_prediction.unsqueeze(-1).unsqueeze(-1)

        Y = self.last_layer(Y)
        Y*= self.presence_prediction.unsqueeze(-1).unsqueeze(-1)
        Y = nn.Upsample(X.shape[-2:])(Y)
        return nn.Softmax(dim=1)(Y)


if __name__ == "__main__":
    device = 'cuda'if torch.cuda.is_available() else 'cpu'
    model = UNet3Plus(in_channels=1).to(device)
    # print(model)
    busi_dataset_path = "Dataset_BUSI_with_GT"

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

    train_df, test_df = train_test_split(dataset_df, train_size=0.8, random_state=42)
    input_shape = (256,256)

    train_ds = BusiDataset(train_df, input_size=input_shape)
    test_ds = BusiDataset(test_df, input_size=input_shape)
    batch_size = 10
    epochs = 150

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
    trainer = Trainer(model,lr=3e-4,batch_size=batch_size, epochs=epochs)
    trainer.train(train_loader,test_loader,"unet3plus_chkpt","unet3plus")
    trainer.save_model("unet3plus_chkpt","unet3plus")
    loss,dice_scores = trainer.test(test_loader)
    print(f"Mean Dice = {np.mean(dice_scores)}")
    image_iter = iter(test_loader)