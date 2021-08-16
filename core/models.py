import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parameter
import torch.nn.functional as F

from backbones_video import resnet_3d, resnext, mobilenet, mobilenetv2, shufflenet, shufflenetv2
from backbones_audio import sincdsnet



class ISRM(nn.Module):
    def __init__(self, inplanes, planes, candidate_speakers=3, stride=1):
        super(ISRM, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.conv_ctx = nn.Conv2d(inplanes, planes, kernel_size=(9,candidate_speakers-1),
                               stride=1, padding=(4,0), bias=False)
        self.bn_ctx   = nn.BatchNorm2d(planes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ref_feat = x[:,:,:,0]
        ctx_feat = x[:,:,:,1:]

        ctx_feat = self.dropout(ctx_feat)
        ctx_feat = self.bn_ctx(self.conv_ctx(ctx_feat)).squeeze(3)
        final_feat = torch.cat([ref_feat, ctx_feat], dim=1)

        return final_feat


class BGRU_Block(nn.Module):
    def __init__(self, inplanes, hidden_units, batch_first=True):
        super(BGRU_Block, self).__init__()
        self.rnn = nn.GRU(inplanes, hidden_units, num_layers=2, dropout=0.1, bidirectional=True, batch_first=batch_first)

    def forward(self, x):
        clip_length = x.size(1)
        x, _ = self.rnn(x)
        x = x[:, clip_length//2, :] # Return only the key-frame index
        return x


class TM_ISRM_Net(nn.Module):
    def __init__(self, num_speakers=3, hidden_units=128):
        super(TM_ISRM_Net, self).__init__()
        self.isrm     = ISRM(512+160, hidden_units, num_speakers)
        self.tm       = BGRU_Block(800, hidden_units, batch_first=True)
        self.fc_final = nn.Linear(128*2, 2)

    def forward(self, x):
        # Inter-Speaker Relation Modeling
        x = self.isrm(x)

        # Temporal Modeling
        x = x.permute(0, 2, 1)
        x = self.tm(x)

        # Final Prediction
        x = self.fc_final(x)

        return x

    
# Audio Network
class Audio_Backbone(nn.Module):
    def __init__(self, backbone, pretrained_path=None):
        super(Audio_Backbone, self).__init__()
        if backbone == 'sincdsnet':
            self.base = sincdsnet.get_model()
        else:
            print("Select and appropriate audio backbone from the list: [sincdsnet]")

        if pretrained_path:
            self.base = load_model(self.base,  pretrained_path)


    def forward(self, a, return_feat=False):
        a = self.base(a) 
        a = a.reshape(a.size(0), -1)
        return a


# Video Network
class Video_Backbone(nn.Module):
    def __init__(self, backbone, pretrained_path=None):
        super(Video_Backbone, self).__init__()
        if backbone == 'resnet18':
            self.base = resnet_3d.generate_model(model_depth=18)
        elif backbone == 'resnext101':
            self.base = resnext.resnext101()
        elif backbone == 'mobilenet':
            self.base = mobilenet.get_model(num_classes=num_classes, sample_size=160, width_mult=2.)
        elif backbone == 'mobilenetv2':
            self.base = mobilenetv2.get_model(num_classes=num_classes, sample_size=160, width_mult=1.)
        elif backbone == 'shufflenet':
            self.base = shufflenet.get_model(groups=3, num_classes=num_classes, width_mult=2.)
        elif backbone == 'shufflenetv2':
            self.base = shufflenetv2.get_model(num_classes=num_classes, sample_size=160, width_mult=2.)
        else:
            print("Select and appropriate video backbone from the list: [resnet18, resnext101, mobilenet, mobilenetv2, shufflenet, shufflenetv2]")
            
        if pretrained_path:
            self.base = load_model(self.base, pretrained_path)
    
        self.avgpool_3d = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, v):
        v = self.base(v) 
        v = self.avgpool_3d(v)
        v = v.reshape(v.size(0), -1)
        return v


class TwoStreamNet(nn.Module):
    def __init__(self, opt):
        super(TwoStreamNet, self).__init__()

        #Audio stream
        self.audio_base = Audio_Backbone(backbone=opt.audio_backbone, pretrained_path=opt.audio_backbone_pretrained_path)

        #Video stream
        self.video_base = Video_Backbone(backbone=opt.video_backbone, pretrained_path=opt.video_backbone_pretrained_path)

        self.relu = nn.ReLU(inplace=True)
        self.fc_128_a = nn.Linear(160, 128)
        self.fc_128_v = nn.Linear(512, 128)

        # Predictions
        self.fc_final = nn.Linear(128*2, 2)
        self.fc_aux_a = nn.Linear(128, 2)
        self.fc_aux_v = nn.Linear(128, 2)


    def forward(self, a, v):
        # Audio Stream
        a = self.audio_base(a)

        #Video Stream
        v = self.video_base(v)

        # Concat Stream Feats
        stream_feats = torch.cat((a, v), 1)

        # Auxiliary supervisions
        a = self.fc_128_a(a)
        a = self.relu(a)
        v = self.fc_128_v(v)
        v = self.relu(v)

        aux_a = self.fc_aux_a(a)
        aux_v = self.fc_aux_v(v)

        # Global supervision
        av = torch.cat((a, v), 1)
        x = self.fc_final(av)

        return x, aux_a, aux_v, stream_feats


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    # state_dict_ = checkpoint
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        # num_batches_tracked need to be filtered out (a version problem
        # see https://stackoverflow.com/questions/53678133/load-pytorch-model-from-0-4-1-to-0-4-0)
        if not (k in state_dict) and 'num_batches_tracked' not in k:
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model