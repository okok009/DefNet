import torch
import torch.nn as nn

from defnet import defnet18, defnet50

class Unet_model(nn.Module):
    def __init__(self, backbone, in_channels_list, num_classes=2, out_channels_list=[64, 128, 256, 512], input_size=400):
        super().__init__()
        self.final = nn.Conv2d(64, num_classes, 1)
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_concat1 = Unet_Up(in_channels_list[0], out_channels_list[0], int(input_size / 2 ** 1))
        self.up_concat2 = Unet_Up(in_channels_list[1], out_channels_list[1], int(input_size / 2 ** 2))
        self.up_concat3 = Unet_Up(in_channels_list[2], out_channels_list[2], int(input_size / 2 ** 3))
        self.up_concat4 = Unet_Up(in_channels_list[3], out_channels_list[3], int(input_size / 2 ** 4))
        self.backbone = backbone

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.backbone(inputs)
        outputs = self.up_concat4(feat4, feat5)
        outputs = self.up_concat3(feat3, outputs)
        outputs = self.up_concat2(feat2, outputs)
        outputs = self.up_concat1(feat1, outputs)
        
        outputs = self.up_conv(outputs)
        outputs = self.final(outputs)
        return outputs
    
    def load_state_dict(self, path):
        w = torch.load(path)
        super().load_state_dict(w)

class Unet_Up(nn.Module):
    def __init__(self, in_channels, out_channels, inputs1_size):
        super(Unet_Up, self).__init__()
        self.up     = nn.UpsamplingBilinear2d(size=inputs1_size)
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) 
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

def unt_defnet18(num_classes=2, pretrained_offical=False, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_offical : true表要載入官方提供的權重

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    if pretrained_offical and pretrained_own is not None:
        raise RuntimeError('(\'pretrained_offical\')和(\'pretrained_own\' is not None)不能同時為true')
    
    if pretrained_offical:
        model = Unet_model(defnet18(pretrained_offical=pretrained_offical), in_channels_list, num_classes=num_classes)
    elif pretrained_own:
        model = Unet_model(defnet18(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(defnet18(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_defnet50(num_classes=2, pretrained_offical=False, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_offical : true表要載入官方提供的權重

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    if pretrained_offical and pretrained_own is not None:
        raise RuntimeError('(\'pretrained_offical\')和(\'pretrained_own\' is not None)不能同時為true')
    
    if pretrained_offical:
        model = Unet_model(defnet50(pretrained_offical=pretrained_offical), in_channels_list, num_classes=num_classes)
    elif pretrained_own:
        model = Unet_model(defnet50(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(defnet50(), in_channels_list, num_classes=num_classes)
    
    return model

if __name__ == '__main__':
    input_test = torch.randn(1,3,400,400)
    model = unt_defnet18(numclass=2, pretrained_offical=True)
    print(len(model(input_test)))