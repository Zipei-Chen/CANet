import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.spectral_norm import spectral_norm
# from spatial_correlation_sampler import spatial_correlation_sample
from collections import OrderedDict
# from .resample2d import Resample2d
# from .DenseNet import DenseNet
# from .DenseNet_real import densenet121
import torchvision.utils as vutils


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


class FlowGen(nn.Module):
    def __init__(self, input_dim=2, dim=64, n_res=2, activ='relu',
                 norm_flow='ln', norm_conv='in', pad_type='reflect', use_sn=True):
        super(FlowGen, self).__init__()

        # self.unet = UNET(3, 3)
        # self.flow_column = FlowColumn(input_dim, dim, n_res, activ,
        #                               norm_flow, pad_type, use_sn)

        # self.conv_column = ConvColumn(input_dim, dim, n_res, activ,
        #                               norm_conv, pad_type, use_sn)
        # self.u_net = DenseUNet()
        # self.u_net2 = DenseUNet2()   # two branch
        # self.u_net3 = DenseUNet3()   # stacked network
        # self.u_net = UNET(3, 3)
        # self.out_select = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, padding=0, stride=1)
        # self.image_generator = generator()

        self.generator1 = new_generator1()
        self.generator2 = new_generator2()

    # def forward(self, inputs, maps):
    # def forward(self, inputs, maps, flow_tensor):
    #     input2_1 = inputs[:, 1:3, :, :]
    #     input2_2 = inputs[:, 4:6, :, :]
    #     input2 = torch.cat([input2_1, input2_2], dim=1)
    #     # input_feature =
    #
    #     # flow_map = self.flow_column(input2)
    #     flow_map = flow_tensor
    #     # flow_map = self.u_net.forward3(inputs)
    #
    #     # flow_map = 1
    #     # images_out = self.conv_column(inputs, flow_map)
    #
    #     images_out, flow_feature = self.u_net(inputs, flow_map, maps)
    #
    #     # smooth = inputs[:, 3:6, :, :]
    #     #
    #     # coarse_image = torch.cat([smooth, images_out], dim=1)
    #     # # coarse_image = torch.cat([inputs, images_out], dim=1)
    #     # images_out2 = self.u_net3(coarse_image, flow_map, maps)
    #
    #     # images_out = self.u_net3(inputs, flow_map, maps)
    #     # flow_feature = None
    #     # images_out2 = self.u_net2(inputs)
    #
    #     # images_out_final = torch.cat([images_out, images_out2], dim=1)
    #     # images_out_final = self.out_select(images_out_final)
    #
    #     # images_out, flow_map, flow_feature = self.u_net.forward4(inputs)
    #     # return images_out, images_out2, flow_map, flow_feature
    #     return images_out, flow_map, flow_feature

    # def forward1(self, inputs, maps):
    # def forward1(self, inputs, maps, flow_tensor):
    #
    #     input2_1 = inputs[:, 1:3, :, :]
    #     input2_2 = inputs[:, 4:6, :, :]
    #     input2 = torch.cat([input2_1, input2_2], dim=1)
    #
    #     # flow_map = self.flow_column(input2)
    #     flow_map = flow_tensor
    #     # flow_map = self.u_net.forward3(inputs)
    #
    #     # flow_map = 1
    #     # images_out = self.conv_column(inputs, flow_map)
    #
    #     # images_out = self.u_net.forward2(inputs, flow_map, maps)
    #
    #     # smooth = inputs[:, 3:6, :, :]
    #
    #     # coarse_image = torch.cat([smooth, images_out], dim=1)
    #     # coarse_image = torch.cat([inputs, images_out], dim=1)
    #     # images_out2 = self.u_net3(coarse_image, flow_map, maps)
    #
    #     images_out = self.u_net3(inputs, flow_map, maps)
    #     # images_out2 = self.u_net2(inputs)
    #
    #     # images_out_final = torch.cat([images_out, images_out2], dim=1)
    #     # images_out_final = self.out_select(images_out_final)
    #
    #     # images_out, flow_map = self.u_net.forward5(inputs)
    #     # return images_out, images_out2, flow_map
    #     return images_out, flow_map

    def feat_extractor(self, inputs, smooths, gts):
        inputs = inputs[:, :1, :, :]
        smooths = smooths[:, :1, :, :]
        gts = gts[:, :1, :, :]

        input = torch.cat((inputs, smooths), dim=1)
        gt = torch.cat((gts, gts), dim=1)

        input_L_feats = self.u_net.feature1(input)
        gt_L_feats = self.u_net.feature1(gt)

        return input_L_feats, gt_L_feats

    def forward(self, input, mask, flow2, flow3, flow4, flow5):
        predict1, predict2, predict3, predict4, predict5, predict6 = self.image_generator(input, mask, flow2, flow3,
                                                                                          flow4, flow5)

        return predict1, predict2, predict3, predict4, predict5, predict6

    def forward1(self, input):
        x = self.generator1(input)

        return x

    def forward2(self, input, mask, flow1, flow2, flow3, flow4):
        f1, f2, f3, f4 = self.generator1.feature_extract1(input)
        l, lab = self.generator2.forward(input, f1, f2, f3, f4, flow1, flow2, flow3, flow4, mask)

        return l, lab


class tripleconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(tripleconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes // 4, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_planes // 4, out_channels=out_planes // 4, kernel_size=3, padding=1,
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=out_planes // 4, out_channels=out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x


class tripleconv_bn(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(tripleconv_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes // 8, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_planes // 8, out_channels=out_planes // 8, kernel_size=3, padding=1,
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=out_planes // 8, out_channels=out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x


class lab_fuse(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(lab_fuse, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class doubleconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(doubleconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


class oneconv(nn.Module):
    def __init__(self, in_planes, out_planes, kernal, padding):
        super(oneconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernal, padding=padding,
                               stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


class oneconv_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernal, padding):
        super(oneconv_relu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernal, padding=padding,
                               stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class deconv(nn.Module):
    def __init__(self, in_planes, out_planes, kernal, stride, padding):
        super(deconv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernal, stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x


class flowcolume(nn.Module):
    def __init__(self):
        super(flowcolume, self).__init__()
        vgg19 = VGG19()
        self.feature1 = vgg19.feature1
        self.feature2 = vgg19.feature2
        self.feature3 = vgg19.feature3
        self.feature4 = vgg19.feature4
        self.feature5 = vgg19.feature5
        self.feature5_conv1 = tripleconv_bn(512, 512)
        self.feature5_conv2 = tripleconv_bn(512, 256)
        self.spatial_attention5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                               padding=1, bias=False)),
            ('conv1', nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        # self.ab_feature5_conv3 = tripleconv(512, 128)
        self.feature5_conv3 = oneconv_relu(256, 32, 3, 1)
        # self.ab_feature5_conv4 = oneconv_relu(128, 64, 3, 1)
        self.predic_5 = oneconv(32, 2, 1, 0)

        self.feature4_conv1 = doubleconv(512, 512)
        self.feature4_conv2 = doubleconv(512, 256)
        self.spatial_attention4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                               padding=1, bias=False)),
            ('conv1', nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        # self.ab_feature4_conv3 = doubleconv(512, 128)
        self.feature4_conv3 = oneconv_relu(256, 32, 3, 1)
        # self.ab_feature4_conv4 = oneconv_relu(128, 64, 3, 1)
        self.predic_4_1 = oneconv(32, 2, 1, 0)
        self.predic_4_2 = oneconv(2, 2, 1, 0)

        self.feature3_conv1 = doubleconv(256, 256)
        self.feature3_conv2 = doubleconv(256, 128)
        # self.ab_feature3_conv3 = doubleconv(256, 128)
        self.spatial_attention3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                               padding=1, bias=False)),
            ('conv1', nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        self.feature3_conv3 = oneconv_relu(128, 32, 3, 1)
        # self.ab_feature3_conv4 = oneconv_relu(128, 64, 3, 1)
        self.predic_3_1 = oneconv(32, 2, 1, 0)
        self.predic_3_2 = oneconv(2, 2, 1, 0)

        self.feature2_conv1 = oneconv_relu(128, 128, 3, 1)
        self.feature2_conv2 = oneconv_relu(128, 128, 3, 1)
        self.spatial_attention2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                               padding=1, bias=False)),
            ('conv1', nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        self.feature2_conv3 = oneconv_relu(128, 32, 3, 1)
        # self.ab_feature2_conv4 = oneconv_relu(64, 64, 3, 1)
        self.predic_2_1 = oneconv(32, 2, 1, 0)
        self.predic_2_2 = oneconv(2, 2, 1, 0)

    def forward(self, x):
        x = self.feature1(x)
        f2 = self.feature2(x)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)

        x = self.feature5_conv1(f5)
        x = self.feature5_conv2(x)
        spatial5 = self.spatial_attention5(x)
        x = x * spatial5
        x = self.feature5_conv3(x)
        flow5 = self.predic_5(x)

        x = self.feature4_conv1(f4)
        x = self.feature4_conv2(x)
        spatial4 = self.spatial_attention4(x)
        x = x * spatial4
        x = self.feature4_conv3(x)
        flow4_ori = self.predic_4_1(x)
        flow4 = self.predic_4_2(flow4_ori + flow5)

        x = self.feature3_conv1(f3)
        x = self.feature3_conv2(x)
        spatial3 = self.spatial_attention3(x)
        x = x * spatial3
        x = self.feature3_conv3(x)
        flow3_ori = self.predic_3_1(x)
        flow3 = self.predic_3_2(flow3_ori + flow4_ori + flow5)

        x = self.feature2_conv1(f2)
        x = self.feature2_conv2(x)
        spatial2 = self.spatial_attention2(x)
        x = x * spatial2
        x = self.feature2_conv3(x)
        flow2_ori = self.predic_2_1(x)
        flow2 = self.predic_2_2(flow2_ori + flow3_ori + flow4_ori + flow5)

        return flow2, flow3, flow4, flow5


class _Transition_up(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, kernal, padding):
        super(_Transition_up, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                           kernel_size=3, stride=1, padding=1, bias=False))

        # self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        self.add_module('upsample',
                        nn.ConvTranspose2d(num_output_features, num_output_features, kernel_size=kernal, stride=2,
                                           padding=padding))

        self.add_module('norm2', nn.BatchNorm2d(num_output_features))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                           kernel_size=3, stride=1, padding=1, bias=False))


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class new_generator1(nn.Module):
    def __init__(self):
        super(new_generator1, self).__init__()
        # self.densenet = densenet121(pretrained=True, progress=True)
        self.densenet = densenet121(pretrained=False, progress=True)
        self.decoder_feature = nn.Sequential()
        trans = _Transition_up(1024, 512, 4, 1)
        self.decoder_feature.add_module('up_sample1', trans)
        trans = _Transition_up(512, 256, 4, 1)
        self.decoder_feature.add_module('up_sample2', trans)
        trans = _Transition_up(256, 128, 4, 1)
        self.decoder_feature.add_module('up_sample3', trans)
        trans = _Transition_up(128, 64, 4, 1)
        self.decoder_feature.add_module('up_sample4', trans)
        predict = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        self.decoder_feature.add_module('predict', predict)

        for params in self.parameters():
            params.requires_grad = False

    def forward(self, input):
        x = self.densenet(input)
        # print(self.decoder_feature)
        x = self.decoder_feature(x)

        return x

    def feature_extract1(self, input):
        x = self.densenet.features.conv0(input)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.relu0(x)
        f1 = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(f1)
        f2 = self.densenet.features.denseblock2(x)
        x = self.densenet.features.transition2(f2)
        f3 = self.densenet.features.denseblock3(x)
        x = self.densenet.features.transition3(f3)
        f4 = self.densenet.features.denseblock4(x)

        return f1, f2, f3, f4


class new_generator2(nn.Module):
    def __init__(self):
        super(new_generator2, self).__init__()
        self.resample2 = Resample2d(kernel_size=2, dilation=1, sigma=1)
        self.resample4 = Resample2d(kernel_size=4, dilation=1, sigma=1)
        # self.resample8 = Resample2d(kernel_size=8, dilation=1, sigma=2)
        # self.resample12 = Resample2d(kernel_size=12, dilation=1, sigma=2)
        self.transition1 = _Transition_up(1024, 512, 4, 1)
        # self.res1 = ResBlock(512, 512)
        self.res1 = ResBlock(1536, 1536)
        self.transition2 = _Transition_up(1536, 512, 4, 1)
        # self.res2 = ResBlock(768, 768)
        self.res2 = ResBlock(1024, 1024)
        self.transition3 = _Transition_up(1024, 256, 4, 1)
        # self.res3 = ResBlock(640, 640)
        self.res3 = ResBlock(512, 512)
        self.transition4 = _Transition_up(512, 256, 4, 1)
        self.res4 = ResBlock(256, 256)

        # self.res4_2 = oneconv_relu(256, 64, 3, 1)
        # self.predict_l = oneconv(64, 1, 1, 0)

        self.res4_2 = oneconv_relu(256, 128, 3, 1)
        self.res4_3 = oneconv_relu(128, 64, 3, 1)
        self.predict_l = oneconv(64, 1, 3, 1)

        self.densenet = densenet121(pretrained=False, progress=True)
        # self.feature_2 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.transition11 = _Transition_up(1024, 512, 4, 1)
        # self.res11 = ResBlock(512, 512)
        self.res11 = ResBlock(1536, 1536)
        self.transition22 = _Transition_up(1536, 512, 4, 1)
        # self.res22 = ResBlock(768, 768)
        self.res22 = ResBlock(1024, 1024)
        self.transition33 = _Transition_up(1024, 256, 4, 1)
        # self.res33 = ResBlock(640, 640)
        self.res33 = ResBlock(512, 512)
        self.transition44 = _Transition_up(512, 256, 4, 1)
        self.res44 = ResBlock(256, 256)

        # self.res44_2 = oneconv_relu(256, 64, 3, 1)
        # self.predict_lab = oneconv(64, 3, 1, 0)

        self.res44_2 = oneconv_relu(256, 128, 3, 1)
        self.res44_3 = oneconv_relu(128, 64, 3, 1)
        self.predict_lab = oneconv(64, 3, 3, 1)

    def forward(self, image, feature1, feature2, feature3, feature4, flow1, flow2, flow3, flow4, mask):
        # transition + concat + res + transition
        _, _, h, w = feature4.size()
        mask4 = F.interpolate(mask, [h, w])
        x = self.resample2(feature4, flow4) * mask4 + feature4 * (1 - mask4)
        x = self.transition1(x)

        _, _, h, w = feature3.size()
        mask3 = F.interpolate(mask, [h, w])
        x2 = self.resample2(feature3, flow3) * mask3 + feature3 * (1 - mask3)
        x = torch.cat([x, x2], dim=1)
        x = self.res1(x)
        x = self.transition2(x)

        _, _, h, w = feature2.size()
        mask2 = F.interpolate(mask, [h, w])
        x2 = self.resample4(feature2, flow2) * mask2 + feature2 * (1 - mask2)
        # x2 = feature2
        x = torch.cat([x, x2], dim=1)
        x = self.res2(x)
        x = self.transition3(x)

        # _, _, h, w = feature1.size()
        # mask1 = F.interpolate(mask, [h, w])
        # x2 = self.resample2(feature1, flow1) * mask1 + feature1 * (1 - mask1)
        x2 = feature1
        x = torch.cat([x, x2], dim=1)
        x = self.res3(x)
        x = self.transition4(x)
        x = self.res4(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        l = self.predict_l(x)

        z = image[:, 1:, :, :]
        x = torch.cat([l, z], dim=1)

        x = self.feature_2(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.relu0(x)
        f1 = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(f1)
        f2 = self.densenet.features.denseblock2(x)
        x = self.densenet.features.transition2(f2)
        f3 = self.densenet.features.denseblock3(x)
        x = self.densenet.features.transition3(f3)
        x = self.densenet.features.denseblock4(x)
        x = self.transition11(x)
        x = torch.cat([x, f3], dim=1)
        x = self.res11(x)
        x = self.transition22(x)
        x = torch.cat([x, f2], dim=1)
        x = self.res22(x)
        x = self.transition33(x)
        x = torch.cat([x, f1], dim=1)
        x = self.res33(x)
        x = self.transition44(x)
        x = self.res44(x)
        x = self.res44_2(x)
        x = self.res44_3(x)
        lab = self.predict_lab(x)

        return l, lab