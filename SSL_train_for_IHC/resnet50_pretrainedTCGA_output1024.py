import torch
from torchvision.models.resnet import Bottleneck, ResNet
from torchsummary import summary
import torch.nn as nn
import torch


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer
        del self.layer4
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)  
        x = self.flatten(x)
        return x


def resnet50_3layers(pretrained, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 0], **kwargs)
    if pretrained:
        #Mingu Kang."Self-supervised pre-trained weights on TCGA," Github, 2023, https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
        file_path = 'path/mocov2_rn50_ep200.torch'
        verbose = model.load_state_dict(
            torch.load(file_path), strict=False
        )
        print("Missing keys:", verbose.missing_keys)   
        print("Unexpected keys:", verbose.unexpected_keys)   
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return model


class PatchEmbed_ResNet50_PretrainedTCGA(nn.Module):
    def __init__(self):
        super(PatchEmbed_ResNet50_PretrainedTCGA, self).__init__()
        self.model = resnet50_3layers(pretrained=True)
        self.num_patches = 0

    def forward(self, x):
        batchsize, _, _, _ = x.size()
        x = self.model(x)
        return x
