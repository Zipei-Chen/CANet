import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math
from .resnext.resnext101_regular import ResNeXt101
from .resample2d import Resample2d
import numpy as np
import pdb


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, activation='lrelu', use_bias=True):
        super(ConvBlock, self).__init__()

        self.norm = nn.BatchNorm2d(output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=use_bias)
        self.conv.apply(weights_init('gaussian'))

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Deconvolution(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, activation='lrelu', use_bias=True):
        super(Deconvolution, self).__init__()

        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=use_bias)
        self.conv.apply(weights_init('gaussian'))

        self.norm = nn.BatchNorm2d(output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
             x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResBlock, self).__init__()

        self.conv = ConvBlock(input_dim, output_dim, 3, 1, 1, 'none', False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        x = x + y
        x = self.lrelu(x)
        return x


class CANet(nn.Module):
    def __init__(self):
        super(CANet, self).__init__()

        self.resnet = ResNeXt101()
        self.conv5 = ConvBlock(1024, 1024, 3, 2, 1, 'lrelu', False)
        self.bottle6 = ConvBlock(1024, 512, 1, 1, 0, 'lrelu', False)
        self.CFT = Resample2d(kernel_size=2, dilation=1, sigma=1)
        self.deconv6 = Deconvolution(1024, 1024, 3, 2, 1, 'lrelu', False)

        # self.bottle7 = ConvBlock(1024, 512, 1, 1, 0, 'lrelu', False)
        # self.resample7 = Resample2d(kernel_size=4, dilation=1, sigma=1)
        self.res7 = ResBlock(2048, 2048)
        self.deconv7 = Deconvolution(2048, 512, 4, 2, 1, 'lrelu', False)

        # self.bottle8 = ConvBlock(512, 256, 1, 1, 0, 'lrelu', False)
        # self.resample8 = Resample2d(kernel_size=4, dilation=1, sigma=1)

        self.res8 = ResBlock(1024, 1024)
        self.deconv8 = Deconvolution(1024, 256, 4, 2, 1, 'lrelu', False)

        self.res9 = ResBlock(512, 512)
        self.deconv9 = Deconvolution(512, 64, 4, 2, 1, 'lrelu', False)

        self.res10 = ResBlock(128, 128)
        self.deconv10 = Deconvolution(128, 64, 4, 2, 1, 'lrelu', False)

        self.predict_l_1 = nn.Conv2d(64, 64, 1, 1, padding=0, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.predict_l_2 = nn.Conv2d(64, 1, 1, 1, padding=0, bias=False)

        self.predict_ab_1 = nn.Conv2d(64, 64, 1, 1, padding=0, bias=False)
        self.predict_ab_2 = nn.Conv2d(64, 2, 1, 1, padding=0, bias=False)

        # stage 2

        self.resnet2 = ResNeXt101()
        self.channel_change = ConvBlock(6, 3, 1, 1, 0, 'none', False)

        self.conv_5 = ConvBlock(1024, 1024, 3, 2, 1, 'lrelu', False)
        self.deconv_6 = Deconvolution(1024, 1024, 3, 2, 1, 'lrelu', False)

        self.res_7 = ResBlock(2048, 2048)
        self.deconv_7 = Deconvolution(2048, 512, 4, 2, 1, 'lrelu', False)

        self.res_8 = ResBlock(1024, 1024)
        self.deconv_8 = Deconvolution(1024, 256, 4, 2, 1, 'lrelu', False)

        self.res_9 = ResBlock(512, 512)
        self.deconv_9 = Deconvolution(512, 64, 4, 2, 1, 'lrelu', False)

        self.res_10 = ResBlock(128, 128)
        self.deconv_10 = Deconvolution(128, 64, 4, 2, 1, 'lrelu', False)

        self.predict = nn.Conv2d(64, 64, 1, 1, padding=0, bias=False)
        self.predict_final = nn.Conv2d(64, 3, 1, 1, padding=0, bias=False)

    def forward(self, x, flow_13_1, flow_13_2, flow_13_3):
        f1, f2, f3, f4 = self.resnet.feature_extractor(x)
        f5 = self.conv5(f4)

        f6_temp = self.bottle6(f5)
        f6_temp = torch.cat([f6_temp, self.CFT(f6_temp, flow_13_1) + self.CFT(f6_temp, flow_13_2) + self.CFT(f6_temp, flow_13_3)], dim=1)
        f6 = self.deconv6(f6_temp)

        # f7_temp = self.bottle6(f4)
        # f7_temp = torch.cat([f7_temp, self.resample7(f7_temp)], dim=1)
        f7 = self.res7(torch.cat([f6, f4], dim=1))
        f7 = self.deconv7(f7)

        # f8_temp = self.bottle8(f3)
        # f8_temp = torch.cat([f8_temp, self.resample8(f8_temp)], dim=1)
        f8 = self.res8(torch.cat([f7, f3], dim=1))
        f8 = self.deconv8(f8)

        f9 = self.res9(torch.cat([f8, f2], dim=1))
        f9 = self.deconv9(f9)

        f10 = self.res10(torch.cat([f9, f1], dim=1))
        f10 = self.deconv10(f10)

        pre_l = self.lrelu(self.predict_l_1(f10))
        pre_l = self.predict_l_2(pre_l)

        pre_ab = self.lrelu(self.predict_ab_1(f10))
        pre_ab = self.predict_ab_2(pre_ab)

        predict_stage1 = torch.cat([pre_l, pre_ab], dim=1)

        input2 = torch.cat([predict_stage1, x], dim=1)

        f1, f2, f3, f4 = self.resnet2.feature_extractor(self.channel_change(input2))
        f5 = self.conv_5(f4)
        # f1:200 200 64     f2:100 100 256      # f3:50 50 512      f4:25 25 1024     f5:12 12 1024

        f6 = self.deconv_6(f5)

        f7 = self.res_7(torch.cat([f6, f4], dim=1))
        f7 = self.deconv_7(f7)

        f8 = self.res_8(torch.cat([f7, f3], dim=1))
        f8 = self.deconv_8(f8)

        f9 = self.res_9(torch.cat([f8, f2], dim=1))
        f9 = self.deconv_9(f9)

        f10 = self.res_10(torch.cat([f9, f1], dim=1))
        f10 = self.deconv_10(f10)

        predict = self.lrelu(self.predict(f10))
        predict_stage2 = self.predict_final(predict)

        return predict_stage1, predict_stage2
