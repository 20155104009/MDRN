'''ResNet-18 Image classfication for cifar-10 with PyTorch

Author 'Sun-qian'.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import PixelUnShuffle

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=4, bias=False,dilation=4),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),

        )
    def forward(self, x):
        y = x
        out = self.left(x)
        out = out + y
        out = F.relu(out)
        return out


class multi_scale_feature(nn.Module):
    def __init__(self,in_channels1 = 1,out_channels1 = 22,in_channels2 = 4,
                 out_channels2 = 84,in_channels3 = 16,out_channels3 = 336,kernel_size=3, stride=1, padding=1, bias=False):
        super(multi_scale_feature, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels2, out_channels2, kernel_size = 3, stride = 1, padding = 1, bias = False),

            nn.ReLU(True),
        )
        self.conv3=nn.Sequential(

            nn.Conv2d(in_channels3, out_channels3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )



    def forward(self,x):
        x1 = self.conv1(x)
        x2 = PixelUnShuffle.pixel_unshuffle(x, 2)
        x2 = self.conv2(x2)
        x2 = F.pixel_shuffle(x2, 2)
        x3 = PixelUnShuffle.pixel_unshuffle(x, 4)
        x3 = self.conv3(x3)
        x3 = F.pixel_shuffle(x3, 4)
        x4 = torch.cat((x1,x2,x3),1)  #1+
        out = x4
        return out




class ResNet(nn.Module):
    def __init__(self,multi_scale_feature,ResidualBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.layer1 = nn.Sequential(multi_scale_feature())
        #self.layer1 = self.make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64,  1, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer5 = self.make_layer(ResidualBlock, 64,  1, stride=1)
        self.layer6 = self.make_layer(ResidualBlock, 64, 1, stride=1)

        self.conv5 = nn.Sequential(

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),

        )
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)



    def forward(self, x):   #batchSize * c * k*w * k*h   128*1*40*40
        y = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.conv5(out)
        return y-out

dn_net = ResNet(multi_scale_feature,ResidualBlock)
print(dn_net)











