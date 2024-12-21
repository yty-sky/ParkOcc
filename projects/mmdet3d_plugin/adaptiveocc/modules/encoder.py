
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
import pdb
import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
import ocnn
from ocnn.modules.resblocks import OctreeResBlock3
import time
from scipy.spatial.transform import Rotation as R

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(OccEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.pc_range = pc_range
        self.fp16_enabled = False
        # self.fp16_enabled = True

    @staticmethod
    def get_reference_points(H, W, Z, bs=1, device='cuda', dtype=torch.float32):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
        return ref_3d

    @staticmethod
    def get_reference_points_octree(octree, depth, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        key = octree.key(depth)
        xs, ys, zs = octree.key2xyz(key, depth)
        xs = (xs + 0.5) / (octree.occ_size[0] * (2 ** depth))
        ys = (ys + 0.5) / (octree.occ_size[1] * (2 ** depth))
        zs = (zs + 0.5) / (octree.occ_size[2] * (2 ** depth))
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.flatten(1)
        bs = 1
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
        return ref_3d


    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):

        cam_extrinsic = []
        cam_intrinsic = []
        poses = []
        for img_meta in img_metas:
            cam_extrinsic.append(img_meta['cam_extrinsic'])
            cam_intrinsic.append(img_meta['cam_intrinsic'])
            poses.append(img_meta['poses'])

        cam_extrinsic = np.asarray(cam_extrinsic)
        cam_extrinsic = reference_points.new_tensor(cam_extrinsic)  # (B, N, 4, 4)

        poses = np.asarray(poses)
        poses = reference_points.new_tensor(poses)

        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        D, B, num_query = reference_points.size()[:3]
        num_cam = cam_extrinsic.size(1)
        temproal_num = poses.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam*temproal_num, 1, 1).unsqueeze(-1)
        transform_matrix = torch.matmul(cam_extrinsic, poses.permute(1,0,2,3)).reshape(1,num_cam*temproal_num, 4, 4)
        transform_matrix = transform_matrix.view(1, B, num_cam*temproal_num, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(transform_matrix.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
        reference_points_cam = reference_points_cam[..., 0:3]

        n = torch.norm(reference_points_cam, dim=-1)
        reference_points_cam[..., 0] = reference_points_cam[..., 0] / n
        reference_points_cam[..., 1] = reference_points_cam[..., 1] / n
        reference_points_cam[..., 2] = reference_points_cam[..., 2] / n

        rotm = R.from_euler('xyz', [np.pi / 2, 0, np.pi / 2]).as_matrix().transpose()
        r = R.from_euler('yzx', [np.pi, np.pi / 2, 0]).as_matrix()
        r = reference_points_cam.new_tensor(np.matmul(r, rotm))[None, None]
        r = r.repeat(D, 1, num_cam*temproal_num, num_query, 1, 1)

        points = torch.matmul(r, reference_points_cam.unsqueeze(-1))
        points = points.squeeze(-1)
        volume_mask = points[..., 2:3] < 0

        norm = torch.norm(points[..., :2], dim=-1)
        theta = torch.arctan(points[..., 2] / norm)
        invnorm = 1.0 / norm
        length_invpol = max(cam_intrinsic[0]['length_invpol'])
        powers = theta.new_tensor(np.arange(0, length_invpol, dtype=np.float32)).view(1, 1, 1, 1, -1)
        theta = theta.unsqueeze(-1)**powers
        pol = theta.new_tensor([[cam_intrinsic[0]['invpol']]])
        pol = pol.unsqueeze(-2).repeat(D, 1, temproal_num, num_query, 1)
        rho = torch.sum(theta * pol, dim=-1)

        x = points[..., 0] * invnorm * rho  # 投影到x方向
        y = points[..., 1] * invnorm * rho  # 投影到y方向

        c = theta.new_tensor([[cam_intrinsic[0]['c']]]).unsqueeze(-1).repeat(D, 1, temproal_num, num_query)
        d = theta.new_tensor([[cam_intrinsic[0]['d']]]).unsqueeze(-1).repeat(D, 1, temproal_num, num_query)
        e = theta.new_tensor([[cam_intrinsic[0]['e']]]).unsqueeze(-1).repeat(D, 1, temproal_num, num_query)
        xc = theta.new_tensor([[cam_intrinsic[0]['xc']]]).unsqueeze(-1).repeat(D, 1, temproal_num, num_query)
        yc = theta.new_tensor([[cam_intrinsic[0]['yc']]]).unsqueeze(-1).repeat(D, 1, temproal_num, num_query)
        proj_x = x * c + y * d + xc
        proj_y = x * e + y + yc

        reference_points_img = torch.stack((proj_x, proj_y), dim=-1)
        reference_points_img[..., 0] /= img_metas[0]['ori_shape'][0][0]
        reference_points_img[..., 1] /= img_metas[0]['ori_shape'][0][1]

        # reference_points_img[..., 0] /= img_metas[0]['img_shape'][0][0]
        # reference_points_img[..., 1] /= img_metas[0]['img_shape'][0][1]

        volume_mask = (volume_mask & (reference_points_img[..., 1:2] > 0.0)
                    & (reference_points_img[..., 1:2] < 1.0)
                    & (reference_points_img[..., 0:1] < 1.0)
                    & (reference_points_img[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            volume_mask = torch.nan_to_num(volume_mask)
        else:
            volume_mask = volume_mask.new_tensor(np.nan_to_num(volume_mask.cpu().numpy()))

        reference_points_img = reference_points_img.permute(2, 1, 3, 0, 4) #num_cam, B, num_query, D, 3
        volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_img, volume_mask

    @auto_fp16()
    def forward(self,
                volume_query,
                key,
                value,
                octree,
                depth,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                *args,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            volume_query (Tensor): Input 3D volume query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        # 主要做：获取参考点和参考像素

        output = volume_query
        intermediate = []

        if depth == -1:
            ref_3d = self.get_reference_points(
                volume_h, volume_w, volume_z, bs=volume_query.size(1), device=volume_query.device,
                dtype=torch.float32)
        else:
            ref_3d = self.get_reference_points_octree(octree, depth, bs=volume_query.size(1),  device=volume_query.device, dtype=volume_query.dtype)

        reference_points_cam, volume_mask = self.point_sampling(ref_3d, self.pc_range, kwargs['img_metas'])
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        volume_query = volume_query.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):

            output = layer(
                volume_query,
                key,
                value,
                octree,
                depth,
                *args,
                ref_3d=ref_3d,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=volume_mask,
                **kwargs)

            volume_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class OccLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 embed_dims,
                 ffn_dropout=0.0,
                 operation_order=None,
                 conv_num=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 resblk_num=2,
                 is_octree=False,
                 occ_size=[50, 50, 4],
                 is_gn=False,
                 **kwargs):
        super(OccLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            embed_dims=embed_dims,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # self.fp16_enabled = True
        self.resblk_num = resblk_num
        self.deblock = nn.ModuleList()
        self.occ_size = occ_size
        self.is_gn = is_gn
        conv_cfg = dict(type='Conv3d', bias=False)
        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        for i in range(conv_num):
            if is_octree == False:
                conv_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1)
                deblock = nn.Sequential(conv_layer,
                                        build_norm_layer(norm_cfg, embed_dims)[1],
                                        nn.ReLU(inplace=True))
                self.deblock.append(deblock)
            else:
                if is_gn:
                # deblock = ocnn.modules.OctreeResBlocks(embed_dims, embed_dims, self.resblk_num, nempty=False)
                    deblock = ocnn.modules.OctreeResBlocks(embed_dims, embed_dims, self.resblk_num, resblk=OctreeResBlock3, group=8, nempty=False)
                    self.deblock.append(deblock)
                else:
                    deblock = ocnn.modules.OctreeResBlocks(embed_dims, embed_dims, self.resblk_num, nempty=False)
                    self.deblock.append(deblock)
        #assert len(operation_order) == 6
        #assert set(operation_order) == set(
        #    ['self_attn', 'norm', 'cross_attn', 'ffn'])

    @auto_fp16()
    def forward(self,
                query,
                key=None,
                value=None,
                octree=None,
                depth=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_3d=None,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in ' f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention todo 改成八叉树卷积
            if layer == 'conv':
                if depth > 0:

                    identity = query
                    query = torch.squeeze(query, 0)
                    for i in range(len(self.deblock)):
                        query = self.deblock[i](query, octree, depth)
                    query = torch.unsqueeze(query, 0)
                    query = query + identity

                else:
                    # bs = query.shape[0]
                    # identity = query
                    # query = query.reshape(bs, self.occ_size[0], self.occ_size[1], self.occ_size[2], -1).permute(0, 4, 1, 2, 3)  # [1, 512, 50, 50, 4]
                    # for i in range(len(self.deblock)):
                    #     query = self.deblock[i](query)
                    # query = query.permute(0, 2, 3, 4, 1).reshape(bs, self.occ_size[0] * self.occ_size[1] * self.occ_size[2], -1)
                    # query = query + identity

                    bs = query.shape[0]
                    identity = query
                    query = query.reshape(bs, volume_h, volume_w, volume_z, -1).permute(0, 4, 1, 2, 3)
                    for i in range(len(self.deblock)):
                        query = self.deblock[i](query)
                    query = query.permute(0, 2, 3, 4, 1).reshape(bs, volume_h * volume_w * volume_z, -1)
                    query = query + identity

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                # query += identity
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1


        return query
