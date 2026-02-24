import torch
import torch.nn as nn


class ResNetClassifier(nn.Module):
    """Custom ResNet for benign/malignant classification on BUSI grayscale images."""

    def __init__(self, in_channel=1, num_classes=2, img_width=256):
        super().__init__()
        self.x7conv3to64   = nn.Conv2d(in_channel, 64, 7, 1, 3, bias=False)
        self.x1conv64to64  = nn.Conv2d(64, 64, 1, 1, bias=False)
        self.x1conv128to64 = nn.Conv2d(128, 64, 1, 1, bias=False)

        self.x3conv64to64   = nn.Conv2d(64,   64,   3, 1, 1, bias=False)
        self.x3conv64to128  = nn.Conv2d(64,   128,  3, 1, 1, bias=False)
        self.x3conv128to64  = nn.Conv2d(128,  64,   3, 1, 1, bias=False)
        self.x3conv128to128 = nn.Conv2d(128,  128,  3, 1, 1, bias=False)
        self.x3conv128to256 = nn.Conv2d(128,  256,  3, 1, 1, bias=False)
        self.x3conv256to128 = nn.Conv2d(256,  128,  3, 1, 1, bias=False)
        self.x3conv256to256 = nn.Conv2d(256,  256,  3, 1, 1, bias=False)
        self.x3conv256to512 = nn.Conv2d(256,  512,  3, 1, 1, bias=False)
        self.x3conv512to256 = nn.Conv2d(512,  256,  3, 1, 1, bias=False)
        self.x3conv512to512 = nn.Conv2d(512,  512,  3, 1, 1, bias=False)
        self.x3conv512to1024  = nn.Conv2d(512,  1024, 3, 1, 1, bias=False)
        self.x3conv1024to512  = nn.Conv2d(1024, 512,  3, 1, 1, bias=False)
        self.x3conv1024to1024 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)

        self.normal64   = nn.BatchNorm2d(64)
        self.normal128  = nn.BatchNorm2d(128)
        self.normal256  = nn.BatchNorm2d(256)
        self.normal512  = nn.BatchNorm2d(512)
        self.normal1024 = nn.BatchNorm2d(1024)
        self.d1normal1024 = nn.BatchNorm1d(1024)

        self.relu    = nn.ReLU(inplace=True)
        self.pool    = nn.MaxPool2d(2)
        self.fc_pool = nn.AvgPool2d(img_width // 16, stride=1)

        self.linear1024to1024 = nn.Linear(1024, 1024)
        self.linear_out       = nn.Linear(1024, num_classes)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        output = self.pool(self.x7conv3to64(x))

        res = self.x1conv64to64(self.x3conv64to64(output))
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

        output = torch.cat([output, res], dim=1)
        output = self.pool(self.x3conv128to128(output))
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        for _ in range(3):
            output = torch.cat([output, res], dim=1)
            output = self.x3conv256to128(output)
            res = output
            output = self.normal128(output)
            output = self.relu(self.x3conv128to128(output))
            output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)
        output = self.pool(self.x3conv256to256(output))
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        for _ in range(5):
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

        for _ in range(2):
            output = torch.cat([output, res], dim=1)
            output = self.x3conv1024to512(output)
            res = output
            output = self.normal512(output)
            output = self.relu(self.x3conv512to512(output))
            output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.pool(self.x3conv1024to1024(output))
        output = self.x3conv1024to1024(output)
        output = self.x3conv1024to1024(output)

        output = self.fc_pool(output)
        output = output.reshape(output.shape[0], -1)

        if output.shape[0] > 1:
            output = self.d1normal1024(output)
        output = self.drop(self.relu(self.linear1024to1024(output)))
        if output.shape[0] > 1:
            output = self.d1normal1024(output)
        output = self.relu(self.linear1024to1024(output))
        output = self.drop(self.relu(self.linear1024to1024(output)))
        if output.shape[0] > 1:
            output = self.d1normal1024(output)
        output = self.linear_out(output)
        return output
