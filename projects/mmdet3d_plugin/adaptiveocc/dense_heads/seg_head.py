import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
import os
from torch.autograd import Variable

class ConvGNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvGNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.gn = nn.GroupNorm(16, out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.gn(feat)
        feat = self.relu(feat)
        return feat


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(SegmentHead, self).__init__()
        self.conv = ConvGNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, size=None):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        if not size is None:
            feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=True)
        return feat


@HEADS.register_module()
class SegHead(nn.Module):
    def __init__(self, n_classes, img_channels, out_channels):
        super(SegHead, self).__init__()
        self.head = SegmentHead(img_channels, out_channels, n_classes)
        self.aux2 = SegmentHead(img_channels, out_channels, n_classes)
        self.aux3 = SegmentHead(img_channels, out_channels, n_classes)
        self.init_weights()

    def forward(self, x, size):

        logits = self.head(x[-1][0], size)
        logits_aux2 = self.aux2(x[-2][0], size)
        logits_aux3 = self.aux3(x[-3][0], size)

        return [logits, logits_aux2, logits_aux3]

    def loss(self, pred_list, gt_mask, weight):

        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
        loss_dict = {}
        gt_mask = gt_mask.squeeze().long()
        for i in range(len(pred_list)):
            pred = pred_list[i]
            loss_occ_i = criterion(pred, gt_mask)
            loss_dict['loss_seg_{}'.format(i)] = weight * loss_occ_i

        return loss_dict


    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)