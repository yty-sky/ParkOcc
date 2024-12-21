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


class DepthHead(nn.Module):
    def __init__(self, in_chan, mid_chan, depth_bin_num):
        super(DepthHead, self).__init__()
        self.conv = ConvGNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_chan, depth_bin_num, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, size=None):
        feat = self.conv(x)
        feat = self.drop(feat)
        if not size is None:
            feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=True)
        feat = self.conv_out(feat)
        return feat


@HEADS.register_module()
class DepHead(nn.Module):
    def __init__(self, depth_bin, depth_bin_corse, min_depth, max_depth, img_channels, out_channels, is_short=True):
        super(DepHead, self).__init__()
        self.depth_bin = depth_bin
        self.depth_bin_corse = depth_bin_corse
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.D = int((max_depth - min_depth) // depth_bin) + 1
        self.corse_D = int((max_depth - min_depth) // depth_bin_corse) + 1
        self.head = DepthHead(img_channels, out_channels, self.D)
        self.corse_head = DepthHead(img_channels, out_channels, self.corse_D)
        self.aux2 = DepthHead(img_channels, out_channels, self.D)
        self.aux3 = DepthHead(img_channels, out_channels, self.D)
        self.is_short = is_short
        self.downsample = 4
        self.init_weights()

    def forward(self, x, size, is_aux=False, is_infer=False):

        if is_infer == True:
            return [self.corse_head(x[-1][0])]
        logits = self.head(x[-1][0], size)
        logits_corse = self.corse_head(x[-1][0])
        if is_aux:
            logits_aux2 = self.aux2(x[-2][0], size)
            logits_aux3 = self.aux3(x[-3][0], size)
            return [logits, logits_aux2, logits_aux3, logits_corse]
        return [logits, logits_corse]

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

    def get_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """

        B, N, H, W = gt_depths.shape
        gt_depths = (gt_depths - self.min_depth) // self.depth_bin
        gt_depths = torch.where((gt_depths < self.D) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))     # (B*N_views, fH, fW)
        gt_depths = gt_depths.long().view(-1)   # (B*N_views*fH*fW, D)
        return gt_depths

    def get_gt_depth_corse(self, gt_depths):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """

        B, N, H, W = gt_depths.shape
        gt_depths = (gt_depths - self.min_depth) // self.depth_bin_corse
        gt_depths = torch.where((gt_depths < self.corse_D) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))     # (B*N_views, fH, fW)
        gt_depths = gt_depths.long().view(-1)   # (B*N_views*fH*fW, D)
        return gt_depths

    @force_fp32()
    def loss(self, depth_preds, depth_maps, weight):
        """
        Args:
            depth_labels: (B, N_views, img_h, img_w)
            depth_preds: (B*N_views, D, fH, fW)
        Returns:

        """
        calss_weight = torch.torch.FloatTensor([4 for i in range(8)] + [1 for i in range(32)]).cuda()
        criterion = nn.CrossEntropyLoss(weight=calss_weight, reduction='mean')

        calss_weight_short = torch.torch.FloatTensor([4 for i in range(4)] + [1 for i in range(16)]).cuda()
        # calss_weight_short = torch.torch.FloatTensor([4 for i in range(8)] + [1 for i in range(32)]).cuda()
        criterion_coarse = nn.CrossEntropyLoss(weight=calss_weight_short, reduction='mean')

        depth_labels = self.get_gt_depth(depth_maps)
        loss_dict = {}
        for i in range(len(depth_preds) - 1):
            # depth_labels = self.get_gt_depth(depth_maps[i])
            fg_mask = depth_maps.view(-1) > 0.0
            depth_label = depth_labels[fg_mask]
            depth_pred = depth_preds[i].permute(0, 2, 3, 1).contiguous().view(-1, self.D)
            depth_pred = depth_pred[fg_mask]
            depth_loss = criterion(depth_pred, depth_label)
            loss_dict['loss_dep_{}'.format(i)] = weight * depth_loss

        B, N, H, W = depth_maps.shape
        gt_depth = depth_maps.view(B * N, H // self.downsample, self.downsample, W // self.downsample, self.downsample, 1)
        gt_depth = gt_depth.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depth = gt_depth.view(-1, self.downsample * self.downsample)
        gt_depth_tmp = torch.where(gt_depth == 0.0, 1e5 * torch.ones_like(gt_depth), gt_depth)
        gt_depth = torch.min(gt_depth_tmp, dim=-1).values
        gt_depth = gt_depth.view(B, N, H // self.downsample, W // self.downsample)
        gt_depth = torch.where(gt_depth >= 1e5, torch.zeros_like(gt_depth), gt_depth)
        depth_labels = self.get_gt_depth_corse(gt_depth)
        fg_mask = gt_depth.view(-1) > 0.0
        depth_label = depth_labels[fg_mask]
        depth_pred = depth_preds[-1].permute(0, 2, 3, 1).contiguous().view(-1, self.corse_D)
        depth_pred = depth_pred[fg_mask]
        depth_loss = criterion_coarse(depth_pred, depth_label)
        loss_dict['loss_dep_coarse'] = weight * depth_loss

        return loss_dict