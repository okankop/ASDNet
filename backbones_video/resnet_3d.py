import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
'''
def conv3x3x3(in_planes, out_planes, temporal_stride=1, spatial_stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(1, dilation, dilation), dilation=(1, dilation, dilation),
                     bias=False)
'''

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=1, spatial_stride=1):
        super().__init__()

        # self.conv1 = conv3x3x3(in_planes, planes, temporal_stride=stride, spatial_stride=spatial_stride, dilation=dilation)
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        # self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.conv2 = conv3x3x3(planes, planes, (stride, 1, 1))  # changed for new
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # for clip 8
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))
                # for dense yowo
                # downsample = nn.Sequential(
                #     conv1x1x1(self.in_planes, planes * block.expansion, (stride, 1, 1)),
                #     nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, layer_wise=False):
        endpoints = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        

        if layer_wise:
            endpoints = []
            x = self.layer1(x)
            endpoints.append(x)
            x = self.layer2(x)
            endpoints.append(x)
            x = self.layer3(x)
            endpoints.append(x)
            x = self.layer4(x)
            endpoints.append(x)
            return endpoints
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x



def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1],   [64, 128, 256, 512], shortcut_type='A')
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2],   [64, 128, 256, 512], shortcut_type='B')
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3],   [64, 128, 256, 512], shortcut_type='A')
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3],   [64, 128, 256, 512], shortcut_type='B')
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3],  [64, 128, 256, 512], shortcut_type='B')
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3],  [64, 128, 256, 512], shortcut_type='B')
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], [64, 128, 256, 512], shortcut_type='B')

    return model


class FPN_R3d(nn.Module):
    '''R50'''
    def __init__(self, model_depth, feature_dim, **kwargs):
        super(FPN_R3d, self).__init__()
        self.feature_dim = feature_dim
        self.base = generate_model(model_depth, **kwargs)
        if model_depth == 50:
            # depths_level = [2048, 1024, 512, 256, 256]
            # temporal_level = [1, 2, 4, 8]
            depths_level = [2048, 1024, 512, 512]
            temporal_level = [2, 4, 8]
            lateral_dim = 256  # 256 for downsample 4 and can use 512 for downsample 8
        elif model_depth == 18:
            # depths_level = [512, 256, 128, 64, 64]
            # temporal_level = [1, 2, 4, 8]
            depths_level = [512, 256, 128, 128]
            # temporal_level = [2, 4, 8, 16]  # input frame 32
            temporal_level = [1, 2, 4, 8]  # input frame 16
            lateral_dim = 128  # 64 for downsample 4 and can use 128 for downsample 8
        else:
            raise NotImplementedError

        self.lateral_dim = lateral_dim
        # Top-down layer
        self.topLayer = nn.Conv2d(depths_level[0], lateral_dim, kernel_size=1, stride=1)

        # Lateral layers (follow slowfast experiments)
        # each output (feature_dim, 1, h, w)
        self.lateral1 = nn.Conv3d(depths_level[1], lateral_dim,
                                  kernel_size=(temporal_level[1], 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.lateral2 = nn.Conv3d(depths_level[2], lateral_dim,
                                  kernel_size=(temporal_level[2], 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        # self.lateral3 = nn.Conv3d(depths_level[3], lateral_dim,
        #                           kernel_size=(temporal_level[3], 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        # self.avgpool = nn.AvgPool3d(kernel_size=(temporal_level[0], 1, 1), stride=1, padding=0)
        for i in range(1, 3):
            # setattr(self, 'upsample_add_{}'.format(i),
            #         UpsampleAdd(depths_level[i], depths_level[i + 1]))
            setattr(self, 'upsample_add_{}'.format(i),
                    UpsampleAdd(lateral_dim, lateral_dim))

    def _init_weights(self, model_path):
        print('=> loading pretrained 3d model {}'.format(model_path))
        pretrained_ckpt = torch.load(model_path)
        self.base.load_state_dict(pretrained_ckpt, strict=False)

    def forward(self, x):
        # Bottom up
        y = self.base(x)  # list [c2,c3,c4,c5]
        # Top down
        p5 = self.topLayer(y[3].squeeze(2))
        # p5 = self.topLayer(self.avgpool(y[3]).squeeze(2))
        p4 = self.upsample_add_1(p5, self.lateral1(y[2]).squeeze(2))
        p3 = self.upsample_add_2(p4, self.lateral2(y[1]).squeeze(2))
        # p2 = self.upsample_add_3(p3, self.lateral3(y[0]).squeeze(2))
        # p2 = p2.squeeze(2)  # (256, H/4, W/4)
        p3 = p3.squeeze(2)

        return p3


class UpsampleAdd(nn.Module):
    ''' Old implementation '''
    def __init__(self, low_dim, high_dim):
        super(UpsampleAdd, self).__init__()
        self.smooth = nn.Conv2d(low_dim, high_dim, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, top_f, lateral_f):
        _, _, H, W = lateral_f.size()
        top_f = F.interpolate(top_f, size=(H, W), mode='bilinear')
        f = self.smooth(top_f + lateral_f)

        return f
