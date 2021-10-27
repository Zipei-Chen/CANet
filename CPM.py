import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


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

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Fullconnection(nn.Module):
    def __init__(self, input_dim, output_dim, activation='lrelu'):
        super(Fullconnection, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, True)

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
        x = self.fc(x)
        if self.activation:
            x = self.activation(x)
        return x


class Correlation_regressor(nn.Module):
    def __init__(self):
        super(Correlation_regressor, self).__init__()

        self.fc1 = Fullconnection(512, 256, 'lrelu')
        self.fc2 = Fullconnection(256, 128, 'lrelu')
        self.fc3 = Fullconnection(128, 1, 'sigmoid')

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class Type_classifier(nn.Module):
    def __init__(self):
        super(Type_classifier, self).__init__()

        self.fc1 = Fullconnection(512, 256, 'lrelu')
        self.fc2 = Fullconnection(256, 128, 'lrelu')
        self.fc3 = Fullconnection(128, 3, 'none')

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CPMmodule(nn.Module):
    def __init__(self):
        super(CPMmodule, self).__init__()

        self.conv1 = ConvBlock(6, 64, 3, 2, 1, 'lrelu', False)
        self.res1 = ResBlock(64, 64)
        self.conv2 = ConvBlock(64, 96, 3, 2, 1, 'lrelu', False)
        self.res2 = ResBlock(96, 96)
        self.conv3 = ConvBlock(96, 96, 3, 2, 1, 'lrelu', False)
        self.res3 = ResBlock(96, 96)
        self.conv4 = ConvBlock(96, 64, 3, 1, 1, 'lrelu', False)
        self.bottle = Fullconnection(1024, 256, 'lrelu')

        self.type_classifier = Type_classifier()
        self.correlation_regressor = Correlation_regressor()

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.conv4(x)
        b, _, _, _ = x.size()
        x = x.view(b, -1)
        x = self.bottle(x)

        y = self.conv1(y)
        y = self.res1(y)
        y = self.conv2(y)
        y = self.res2(y)
        y = self.conv3(y)
        y = self.res3(y)
        y = self.conv4(y)
        y = y.view(b, -1)
        y = self.bottle(y)

        patch_type = self.type_classifier(torch.cat([x, y], dim=1))
        correlation_degree = self.correlation_regressor(torch.cat([x, y], dim=1))

        return patch_type, correlation_degree