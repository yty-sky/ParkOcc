import copy
import os.path

import numpy as np
from torch.utils.data import Dataset
import mmcv
from os import path as osp
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torch.utils.data import Dataset
import random
import collections
from mmcv.parallel import DataContainer as DC

class RayDataset(Dataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, occ_size, pc_range, data_root, pred_dir, temproal_num=8, classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.class_names = classes
        self.temproal_num = temproal_num
        self.data_infos = mmcv.load(data_root)['infos']
        self.pred_dir = pred_dir

    def __getitem__(self, index):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            data_id=index,
            scene_id=info['scene_id'],
            frame_id=info['frame_idx'],
            occ_path=info['occ_path'],
            pred_path=os.path.join(self.pred_dir, f"{info['scene_id']}/occ_pred_{info['frame_idx']}.npy"),
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
        )

        output_origin_list = collections.deque()
        pose_init = np.array(info['pose']).reshape(4, 4)
        output_origin_list.append(np.zeros(3))

        left_index = 1
        right_index = 1
        scene_id = info['scene_id']
        while left_index > 0 or right_index > 0:
            if left_index > 0:
                left = index - left_index
                if left < 0:
                    left_index = 0
                elif self.data_infos[left]['scene_id'] != scene_id:
                    left_index = 0
                else:
                    pose = np.array(self.data_infos[left]['pose']).reshape(4, 4)
                    origin_tf = np.linalg.inv(pose_init) @ pose
                    if np.abs(origin_tf[0][3]) < 10 and np.abs(origin_tf[1][3]) < 10:
                        output_origin_list.appendleft(origin_tf[:3, 3])
                        left_index += 1
                    else:
                        left_index = 0

            if right_index > 0:
                right = index + right_index
                if right >= len(self.data_infos):
                    right_index = 0
                elif self.data_infos[right]['scene_id'] != scene_id:
                    right_index = 0
                else:
                    pose = np.array(self.data_infos[right]['pose']).reshape(4, 4)
                    origin_tf = np.linalg.inv(pose_init) @ pose
                    if np.abs(origin_tf[0][3]) < 10 and np.abs(origin_tf[1][3]) < 10:
                        output_origin_list.append(origin_tf[:3, 3])
                        right_index += 1
                    else:
                        right_index = 0

        if len(output_origin_list) > self.temproal_num:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, self.temproal_num)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))

        input_dict.update(
            dict(
                poses=output_origin_tensor,
            ))

        # return input_dict
        data = {}
        data['img_metas'] = DC(input_dict, cpu_only=True)
        return data

    def __len__(self):
        return len(self.data_infos)

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {0: 'IoU'}
            class_num = len(self.class_names) + 1
            for i, name in enumerate(self.class_names):
                class_names[i + 1] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0)
            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])


        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict={'Acc':results[0],
                          'Comp':results[1],
                          'CD':results[2],
                          'Prec':results[3],
                          'Recall':results[4],
                          'F-score':results[5]}

        return results_dict
