import torch
import torch.nn as nn

class resNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1) -> None:
        super(resNet, self).__init__()
        # input img 224X224
        self.x7conv3to64 = nn.Conv2d(3,64,7,1,1,bias=False)

        self.x3conv64to64 = nn.Conv2d(64,64,3,1,1,bias=False)
        self.x3conv64to128 = nn.Conv2d(64,128,3,1,1,bias=False)
        self.x3conv128to128 = nn.Conv2d(128,128,3,1,1,bias=False)
        self.x3conv128to256 = nn.Conv2d(128,256,3,1,1,bias=False)
        self.x3conv256to256 = nn.Conv2d(256,256,3,1,1,bias=False) 
        self.x3conv256to512 = nn.Conv2d(256,512,3,1,1,bias=False) 
        self.x3conv512to512 = nn.Conv2d(512,512,3,1,1,bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2)
        self.fc_pool = nn.AvgPool2d(14, stride=1)

    
    def forward(self, x):
        output = self.x7conv3to64(x)
        output = self.pool(output)
        for i in range(0,6):
            output = self.x3conv64to64(output)
        output = self.x3conv64to128(output)
        output = self.pool(output)
        for i in range(0,7):
            output = self.x3conv128to128(output)
        output = self.x3conv128to256(output)
        output = self.pool(output)
        for i in range(0,11):
            output = self.x3conv256to256(output)
        output = self.x3conv256to512(output)
        output = self.pool(output)
        for i in range(0,5):
            output = self.x3conv512to512(output)
        
        