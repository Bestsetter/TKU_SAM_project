import torch
import torch.nn as nn

class resNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1,img_width=224, img_heigth=224):# input img 224X224
        super(resNet, self).__init__() 
        
        self.x7conv3to64 = nn.Conv2d(in_channel,64,7,1,1,bias=False)
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

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2)
        self.fc_pool = nn.AvgPool2d(14, stride=1)

        self.linear1024to1024 = nn.Linear(1024, 1024)
        self.linear1024to2 = nn.Linear(1024, 2)
        self.drop = nn.Dropout(0.1)

        # torch.cat([x, y], dim=1)
        

    
    def forward(self, x):
        output = self.x7conv3to64(x)
        output = self.pool(output)
        print(output.shape)
        res = self.x3conv64to64(output)
        res = self.x1conv64to64(res)
        print(output.shape, res.shape)
        output = self.relu(self.x3conv64to64(output))
        output = self.relu(self.x3conv64to64(output))
        
        output = torch.cat([output, res], dim=1)
        res = self.x1conv128to64(output)
        output = self.relu(self.x3conv128to64(output))
        output = self.relu(self.x3conv64to64(output))

        output = torch.cat([output, res], dim=1)
        res = self.x1conv128to64(output)
        output = self.relu(self.x3conv128to64(output))
        output = self.relu(self.x3conv64to64(output))

        output = torch.cat([output, res], dim=1)  # 1
        output = self.x3conv128to128(output)
        output = self.pool(output)
        res = output
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1) # 2
        output = self.x3conv256to128(output)
        res = output
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 3
        output = self.x3conv256to128(output)
        res = output
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 4
        output = self.x3conv256to128(output)
        res = output
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 1
        output = self.x3conv256to256(output)
        output = self.pool(output)
        res = output
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))
        
        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to512(output)
        res = output
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to512(output)
        res = output
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to512(output)
        res = output
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to1024(output)
        output = self.pool(output)
        print('4.4', output.shape)
        output = self.fc_pool(output)
        print('4.5', output.shape)
        output = output.reshape(output.shape[0], -1)
        print('5', output.shape)
        output = self.drop(self.linear1024to1024(output))
        output = self.relu(self.linear1024to1024(output))
        output = self.drop(self.linear1024to1024(output))
        output = self.relu(self.linear1024to2(output))
        print('6', output.shape)
        return output
        
x = torch.randn(10,3,228,228)
model = resNet()
print('in', x.shape)
print('m',model(x).shape)

# x = torch.randn(10,3,224,224)
# y = torch.randn(10,3,224,224)
# print(torch.cat([x, y], dim=1).shape)

