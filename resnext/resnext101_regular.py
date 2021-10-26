import torch
from torch import nn

#from resnext import resnext_101_32x4d_
#from resnext.config import resnext_101_32_path
from . import resnext_101_32x4d_,resnext_101_32x4d_fork_
from .config import resnext_101_32_path


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        net.load_state_dict(torch.load(resnext_101_32_path))
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])                   # 200 * 200 * 64
        self.layer1 = nn.Sequential(*net[3: 5])                 # 100 * 100 * 256
        self.layer2 = net[5]                                    # 50 * 50 * 512
        self.layer3 = net[6]                                    # 25 * 25 * 1024

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer3

    def feature_extractor(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        return layer0, layer1, layer2, layer3


class ResNeXt101_Fork(nn.Module):
    def __init__(self):
        super(ResNeXt101_Fork, self).__init__()
        net = resnext_101_32x4d_fork_.resnext_101_32x4d_fork
        net.load_state_dict(torch.load(resnext_101_32_path))
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]


    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer3
