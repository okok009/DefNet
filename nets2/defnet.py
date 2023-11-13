import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets

class Def_Block(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, downsampling=False):
        super().__init__()
        if downsampling:
            self.offset = nn.Conv2d(inplanes, 2*9, 3, 2, 1)
            self.dconv = torchvision.ops.DeformConv2d(inplanes, planes*4, 3, 2, 1)

            self.conv1 = nn.Conv2d(inplanes, planes, 3, 2, 1)
        else:
            self.offset = nn.Conv2d(inplanes, 2*9, 3, 1, 1)
            self.dconv = torchvision.ops.DeformConv2d(inplanes, planes*4, 3, 1, 1)

            self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*4, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        offset = self.offset(input)
        dconv_x = self.dconv(input, offset)

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = (dconv_x + x)/2

        return x

class Def_Block_S(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, downsampling=False):
        super().__init__()
        if downsampling:
            self.conv1 = nn.Conv2d(inplanes, planes, 3, 2, 1)
        else: 
            self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.offset = nn.Conv2d(planes, 2*9, 3, 1, 1)
        self.dconv = torchvision.ops.DeformConv2d(planes, planes, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        offset = self.offset(x)
        x = self.dconv(x, offset)

        return x

class DefNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks):
        downsampling = False
        if planes != 64:
            downsampling = True
        layers = []
        layers.append(block(self.inplanes, planes, downsampling))
        self.inplanes = planes * block.expansion
        downsampling = False
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, downsampling))
        return nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x1 = self.relu(x)

        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x1, x2, x3, x4, x5]


def defnet50(pretrained=False):
    model = DefNet(Def_Block, [3, 4, 6, 3])

    return model

def defnet18(pretrained=False):
    model = DefNet(Def_Block_S, [2, 2, 2, 1])

    return model