# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.adaptiveocc.apis.test import custom_multi_gpu_test, custom_single_gpu_test, custom_single_gpu_test_ray, custom_multi_gpu_test_ray
import os.path as osp
import numpy as np
import torch.nn as nn
from projects.mmdet3d_plugin.datasets.ray_metrics import main_single_rayiou
from projects.mmdet3d_plugin.datasets.ray_dataset import RayDataset
from prettytable import PrettyTable
from torch.utils.data.distributed import DistributedSampler

pc_range = [-10, -10, -2.6, 10, 10, 0.6]
occ_size = [200, 200, 32]
occ_class_names = ['car', 'wall', 'road', 'park', 'person']
data_root = "./data/park_infos_val_8_29.pkl"
pred_dir = "./save_res/AdaptiveOcc"
txt_dir = "./txt_dir/lr_0.3_sem_depth_forward.txt"
_is_octree = True
# _is_octree = False

class Ray_Model(nn.Module):
    def __init__(self):
        super(Ray_Model, self).__init__()
        self.occ_class_names = occ_class_names
        self.thresholds = [0.2, 0.4, 0.8]
        self.gt_cnt = np.zeros([len(self.occ_class_names)])
        self.pred_cnt = np.zeros([len(self.occ_class_names)])
        self.tp_cnt = np.zeros([len(self.thresholds), len(self.occ_class_names)])

        self.thresholds_short = [0.2]
        self.gt_cnt_short = np.zeros([len(self.occ_class_names)])
        self.pred_cnt_short = np.zeros([len(self.occ_class_names)])
        self.tp_cnt_short = np.zeros([len(self.thresholds_short), len(self.occ_class_names)])

        # just pass
        self.head = nn.Linear(2, 1)

    def forward(self, occ_pred, occ_gt, lidar_origin):
        tp_cnt, gt_cnt, pred_cnt, tp_cnt_short, gt_cnt_short, pred_cnt_short = main_single_rayiou(occ_pred, occ_gt, lidar_origin, self.occ_class_names)
        return tp_cnt, gt_cnt, pred_cnt, tp_cnt_short, gt_cnt_short, pred_cnt_short


    def evaulate(self, tp_cnt_list, gt_cnt_list, pred_cnt_list, tp_cnt_short_list, gt_cnt_short_list, pred_cnt_short_list):

        for i in range(len(tp_cnt_list)):
            self.tp_cnt += tp_cnt_list[i]
            self.gt_cnt += gt_cnt_list[i]
            self.pred_cnt += pred_cnt_list[i]
            self.tp_cnt_short += tp_cnt_short_list[i]
            self.gt_cnt_short += gt_cnt_short_list[i]
            self.pred_cnt_short += pred_cnt_short_list[i]


        iou_list = []
        iou_list_short = []
        for j, threshold in enumerate(self.thresholds):
            iou_list.append((self.tp_cnt[j] / (self.gt_cnt + self.pred_cnt - self.tp_cnt[j])))

        for j, threshold in enumerate(self.thresholds_short):
            iou_list_short.append((self.tp_cnt_short[j] / (self.gt_cnt_short + self.pred_cnt_short - self.tp_cnt_short[j])))

        rayiou = np.nanmean(iou_list)
        rayiou_0 = np.nanmean(iou_list[0])
        rayiou_1 = np.nanmean(iou_list[1])
        rayiou_2 = np.nanmean(iou_list[2])

        rayiou_short = np.nanmean(iou_list_short)

        table = PrettyTable([
            'Class Names',
            'RayIoU@0.2', 'RayIoU@0.4', 'RayIoU@0.8', 'RayIoUShort@0.2'
        ])
        table.float_format = '.3'

        for i in range(len(self.occ_class_names)):
            table.add_row([
                self.occ_class_names[i],
                iou_list[0][i], iou_list[1][i], iou_list[2][i], iou_list_short[0][i]
            ], divider=(i == len(self.occ_class_names) - 1))

        table.add_row(['MEAN', rayiou_0, rayiou_1, rayiou_2, rayiou_short])

        print(table)

        torch.cuda.empty_cache()

        return {
            'RayIoU': rayiou,
            'RayIoU@0.2': rayiou_0,
            'RayIoU@0.4': rayiou_1,
            'RayIoU@0.8': rayiou_2,
            'RayIoUShort@0.2': rayiou_short,
        }

    # def record(self, iou):
    #     with open(txt_dir, 'w') as file:
    #         for i in range(len(iou)):
    #             scene_frame = iou[i][0]
    #             iou_res = iou[i][1]
    #             file.write(f"{i}:  {scene_frame}  {iou_res[0]:.3f}  {iou_res[1]:.3f}  {iou_res[2]:.3f}  {iou_res[3]:.3f}  {iou_res[4]:.3f}\n")



def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    torch.random.manual_seed(0)
    np.random.seed(0)

    # build the dataloader
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher)

    dataset = val_dataset = RayDataset(occ_size, pc_range, data_root, pred_dir, temproal_num=8, classes=occ_class_names)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=True,
        shuffle=False,
        nonshuffler_sampler=dict(type='DistributedSampler'),
    )

    model = Ray_Model()

    if not distributed:
        model = MMDataParallel(model.cuda())
        tp_cnt_list, gt_cnt_list, pred_cnt_list, tp_cnt_short_list, gt_cnt_short_list, pred_cnt_short_list, iou = custom_single_gpu_test_ray(model, data_loader, is_octree=_is_octree)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        tp_cnt_list, gt_cnt_list, pred_cnt_list, tp_cnt_short_list, gt_cnt_short_list, pred_cnt_short_list, iou = custom_multi_gpu_test_ray(model, data_loader, is_octree=_is_octree)

    rank, _ = get_dist_info()
    if rank == 0:
        model.module.evaulate(tp_cnt_list, gt_cnt_list, pred_cnt_list, tp_cnt_short_list, gt_cnt_short_list, pred_cnt_short_list)
        # model.module.record(iou)


if __name__ == '__main__':
    main()



# | Class Names | RayIoU@0.25 | RayIoU@0.5 | RayIoU@1 |
# +-------------+-------------+------------+----------+
# |     car     |    0.243    |   0.427    |  0.622   |
# |     wall    |    0.176    |   0.309    |  0.458   |
# |     road    |    0.443    |   0.671    |  0.827   |
# |     park    |    0.227    |   0.408    |  0.580   |
# |    person   |    0.146    |   0.249    |  0.318   |
# +-------------+-------------+------------+----------+
# |     MEAN    |    0.247    |   0.413    |  0.561   |
# +-------------+-------------+------------+----------+
