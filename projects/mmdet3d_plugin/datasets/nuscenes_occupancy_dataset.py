import copy
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
import pdb, os
from projects.mmdet3d_plugin.datasets.image_utils import CamModel
import random
import collections
@DATASETS.register_module()
class CustomNuScenesOccDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, occ_size, pc_range, build_octree, build_octree_up, temproal_num=1, is_seg=True, is_depth=True,
                 is_train=True, use_semantic=False, classes=None, overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.build_octree = build_octree
        self.build_octree_up = build_octree_up
        self.use_semantic = use_semantic
        self.class_names = classes
        self._set_group_flag()
        self.temproal_num = temproal_num
        self.temproal_info = []
        self.is_seg = is_seg
        self.is_depth = is_depth
        self.is_train = is_train
        # self.calculate_temproal(2,4)
        # if self.is_train:
        #     self.calculate_temproal(0.25, 0.75)
        # else:
        #     self.calculate_temproal(0.25, 0.75)

    # def calculate_temproal(self, time_min, time_max):
    #
    #     for i, data in enumerate(self.data_infos):
    #         # 小范围遍历
    #         j = i
    #         scene_id = data['scene_id']
    #         timestamp = data['timestamp']
    #         temproal_list = []
    #         while True:
    #             time_dif = timestamp - self.data_infos[j]['timestamp']
    #             if time_dif >= time_min and time_dif <= time_max:
    #                 temproal_list.append(j)
    #
    #             j -= 1
    #             if j < 0 or self.data_infos[j]['scene_id'] != scene_id:
    #                 self.temproal_info.append(temproal_list)
    #                 break
    #             if time_dif > time_max:
    #                 self.temproal_info.append(temproal_list)
    #                 break

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            data_id=index,
            scene_id=info['scene_id'],
            frame_id=info['frame_idx'],
            occ_path=info['occ_path'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            build_octree = np.array(self.build_octree),
            build_octree_up = self.build_octree_up
        )
        if self.is_depth and self.is_train:
            input_dict.update(dict(pts_path=info['pts_path']))

        if self.modality['use_camera']:

            image_paths = []
            sem_paths = []
            cam_extrinsic = []
            cam_intrinsics = {'length_invpol': [], 'invpol': [], 'c': [], 'd': [], 'e': [], 'xc': [], 'yc': []}
            cam_intrinsics.update({'length_pol': [], 'pol': []})
            poses = []

            pose_init = np.array(info['pose']).reshape(4, 4)
            poses.append(np.eye(4))
            # poses.append(pose_init)

            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                if self.is_seg:
                    sem_paths.append(cam_info['sem_path'])

                intrinsic_path = cam_info['cam_intrinsic']
                cam_model = CamModel(intrinsic_path)

                cam_intrinsics['length_invpol'].append(cam_model.length_invpol)
                cam_intrinsics['invpol'].append(cam_model.invpol)
                cam_intrinsics['length_pol'].append(cam_model.length_pol)
                cam_intrinsics['pol'].append(cam_model.pol)
                cam_intrinsics['c'].append(cam_model.c)
                cam_intrinsics['d'].append(cam_model.d)
                cam_intrinsics['e'].append(cam_model.e)
                cam_intrinsics['xc'].append(cam_model.xc)
                cam_intrinsics['yc'].append(cam_model.yc)

                extrinsic_path = cam_info['cam_extrinsic']
                extrinsic = np.genfromtxt(extrinsic_path, delimiter=',')
                cam_extrinsic.append(extrinsic)

            # count = 1
            # temproal_img_index = index
            # while count < self.temproal_num:
            #     if len(self.temproal_info[temproal_img_index]) != 0:
            #         if self.is_train:
            #             temproal_img_index = random.choice(self.temproal_info[temproal_img_index])
            #         else:
            #             temproal_img_index = self.temproal_info[temproal_img_index][0]
            #         info = self.data_infos[temproal_img_index]
            #     for cam_type, cam_info in info['cams'].items():
            #         image_paths.append(cam_info['data_path'])
            #
            #     pose = np.array(info['pose']).reshape(4,4)
            #     pose = np.linalg.inv(pose) @ pose_init
            #     poses.append(pose)
            #     count += 1

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    cam_intrinsic=cam_intrinsics,
                    cam_extrinsic=cam_extrinsic,
                    poses=poses,
                ))

            if self.is_seg:
                input_dict.update(
                    dict(sem_filename = sem_paths)
                )

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            info = self.data_infos[idx]
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

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
