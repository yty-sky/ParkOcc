# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

#import open3d as o3d
from tkinter.messagebox import NO
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.datasets.evaluation_metrics import evaluation_octree_semantic_park, evaluation_semantic_park, evaluation_octree_semantic_park_merge, octree_to_voxel
from sklearn.metrics import confusion_matrix as CM
import time, yaml, os
import torch.nn as nn
import pdb
import ocnn
from ocnn.octree import Octree
from ocnn.octree.shuffled_key import xyz2key, key2xyz
from mmdet.models import builder
import numpy as np
import torch.nn.functional as F

@DETECTORS.register_module()
class AdaptiveOcc(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 seg_head=None,
                 dep_head=None,
                 img_view_transformer=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_semantic=True,
                 is_vis=False,
                 is_seg=True,
                 is_depth=True,
                 use_depth=True,
                 is_aux=True,
                 temproal_num=1,
                 version='v1',
                 ):

        super(AdaptiveOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        if seg_head:
            self.seg_head = builder.build_head(seg_head)
        if dep_head:
            self.dep_head = builder.build_head(dep_head)
        if img_view_transformer:
            self.img_view_transformer = builder.build_neck(img_view_transformer)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = True
        self.use_semantic = use_semantic
        self.is_vis = is_vis
        self.is_seg = is_seg
        self.is_depth = is_depth
        self.is_aux = is_aux
        self.use_depth = use_depth
        self.temproal_num = temproal_num

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)  # {[6, 256, 232, 400], [6, 512, 116, 200], [6, 1024, 58, 100], [6, 2048, 29, 50]}
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped


    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_occ,
                          img_metas,
                          occ_feat=None):

        octree = Octree(depth=4, pc_range=[-10, -10, -2.6, 10, 10, 0.6], occ_size=[25, 25, 4]) # new fast version
        octree.cuda()
        octree.build_octree_aux(torch.squeeze(gt_occ, 0), img_metas[0]['build_octree'], img_metas[0]['build_octree_up'], 6)   # gt_occ:

        outs, _, outs_aux = self.pts_bbox_head(pts_feats, img_metas, octree, occ_feat)
        loss_inputs = [octree, outs, outs_aux]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img'))
    def forward_train(self,
                      img_metas=None,
                      gt_occ=None,
                      img=None,
                      mask=None,
                      depth_map=None,
                      occ_feat=None
                      ):

        img_feats = self.extract_feat(img=img, img_metas=img_metas)  # {(1, 6, 256, 232, 400), (1, 6, 256, 116, 200), (1, 6, 256, 58, 100), (1, 6, 256, 29, 50)}三层特征 -> 四层特征

        losses = dict()
        if self.is_seg:
            size = mask.size()[2:]
            outs = self.seg_head(img_feats, size)
            losses_seg = self.seg_head.loss(outs, mask, 5)
            losses.update(losses_seg)

        if self.is_depth:
            size = depth_map.size()[2:]
            is_aux=True
            outs = self.dep_head(img_feats, size, is_aux=is_aux)
            depth_list = depth_map
            losses_dep = self.dep_head.loss(outs, depth_list, 5)
            losses.update(losses_dep)

            occ_feat = None
            if self.use_depth:
                occ_feat, depth = self.img_view_transformer(img_feats[-1], outs[-1], img_metas)

        losses_pts = self.forward_pts_train(img_feats, gt_occ, img_metas, occ_feat)
        losses.update(losses_pts)

        return losses


    def forward_test(self, img_metas, img=None, gt_occ=None, **kwargs):

        if gt_occ is not None:
            octree = Octree(depth=4, pc_range=[-10, -10, -2.6, 10, 10, 0.6], occ_size=[25, 25, 4]) # new fast version
            octree = octree.cuda()
            octree.build_octree_aux(torch.squeeze(gt_occ, 0), img_metas[0]['build_octree'], img_metas[0]['build_octree_up'], 6)
        else:
            octree = None

        octree_gt = octree
        pred_occ, pred_octree = self.simple_test(img_metas, img, **kwargs)

        if self.is_vis:
            self.generate_output(pred_occ, img_metas, pred_octree, octree_gt)

        if self.use_semantic:
            class_num = 6
            self.generate_output_pkl(pred_occ, pred_octree, img_metas)
            eval_results = evaluation_octree_semantic_park(pred_occ, pred_octree, gt_occ, img_metas[0], class_num)

        return {'evaluation': eval_results}

    def simple_test_pts(self, x, img_metas, rescale=False, occ_feat=None):
        """Test function"""
        outs, octree, _, = self.pts_bbox_head(x, img_metas, occ_feat=occ_feat)
        return outs, octree

    @auto_fp16(apply_to=('img'))
    def simple_test(self, img_metas, img=None, rescale=False, occ_feat=None):
        """Test function without augmentaiton."""

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.is_depth:
            is_aux=False
            outs = self.dep_head(img_feats, size=None, is_aux=is_aux, is_infer=True)

            if self.use_depth:
                occ_feat, depth = self.img_view_transformer(img_feats[-1], outs[-1], img_metas)

        output, octree = self.simple_test_pts(img_feats, img_metas, rescale=rescale, occ_feat=occ_feat)

        return output, octree

    def generate_output(self, pred_occ, img_metas, octree, octree_gt):

        image_filename = str(img_metas[0]['scene_id']) + '_' + img_metas[0]['occ_path'].replace('.npy', '').split('/')[-1]
        save_dir = os.path.join('visual_dir', image_filename)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(4):
            _, pred = torch.max(torch.softmax(pred_occ[i], dim=1), dim=1)
            key = octree.keys[i]
            x, y, z = octree.key2xyz(key, i)
            occ = torch.stack([x, y, z, pred], dim=1)
            occ = occ.cpu().numpy()
            np.save(os.path.join(save_dir, f'pred_{i}.npy'), occ)

            if octree_gt is not None:
                key = octree_gt.keys[i]
                x, y, z = octree_gt.key2xyz(key, i)
                gt = octree_gt.gt[i]
                occ = torch.stack([x, y, z, gt], dim=1)
                occ = occ.cpu().numpy()
                np.save(os.path.join(save_dir, f'gt_{i}.npy'), occ)

    def generate_output_pkl(self, pred_occ, pred_octree, img_metas):

        image_filename_dir = f"{img_metas[0]['scene_id']}"
        save_dir = os.path.join('save_res/AdaptiveOcc/', image_filename_dir)
        os.makedirs(save_dir, exist_ok=True)
        image_filename = os.path.join(save_dir, f"occ_pred_{img_metas[0]['frame_id']}.npy")

        pred_volume = torch.zeros((200, 200, 32), dtype=torch.int64)
        pred_volume = pred_volume.cuda()

        for i in range(4):
            _, pred = torch.max(torch.softmax(pred_occ[i], dim=1), dim=1)
            mask = (pred != 0) * (pred != 6)
            pred = pred[mask]
            key = pred_octree.keys[i][mask]
            x, y, z = pred_octree.key2xyz(key, i)
            pred_value = torch.stack([x, y, z, pred], dim=1)
            pred_volume = octree_to_voxel(pred_value, pred_volume, i)

        np.save(image_filename, pred_volume.cpu().numpy())


    def get_downsampled_gt_depth(self, gt_depth_init, downsample_list):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """
        B, N, H, W = gt_depth_init.shape
        # (B*N_views, fH, downsample, fW, downsample, 1)
        gt_depths_list = []
        for downsample in downsample_list:
            gt_depth = gt_depth_init.view(B * N, H // downsample, downsample, W // downsample, downsample, 1)
            # (B*N_views, fH, fW, 1, downsample, downsample)
            gt_depth = gt_depth.permute(0, 1, 3, 5, 2, 4).contiguous()
            # (B*N_views*fH*fW, downsample, downsample)
            gt_depth = gt_depth.view(-1, downsample * downsample)
            gt_depth_tmp = torch.where(gt_depth == 0.0, 1e5 * torch.ones_like(gt_depth), gt_depth)
            gt_depth = torch.min(gt_depth_tmp, dim=-1).values
            # (B*N_views, fH, fW)
            gt_depth = gt_depth.view(B, N, H // downsample, W // downsample)
            gt_depth = torch.where(gt_depth >= 1e5, torch.zeros_like(gt_depth), gt_depth)
            gt_depths_list.append(gt_depth)
        return gt_depths_list