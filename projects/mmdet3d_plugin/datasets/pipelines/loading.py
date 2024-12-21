#import open3d as o3d
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

distance_z = 8
distance_low1 = 70
distance_high1 = 130

distance_low2 = 40
distance_high2 = 160

@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class. 
    """

    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic

    
    def __call__(self, results):
        occ = np.load(results['occ_path'])
        occ = occ.astype(np.float32)
        
        # class 0 is 'ignore' class
        # if self.use_semantic:
        #     occ[..., 3][occ[..., 3] == 0] = 255
        # else:
        #     occ = occ[occ[..., 3] > 0]
        #     occ[..., 3] = 1
        
        results['gt_occ'] = occ
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadSemantic(object):
    def __init__(self, size, downsample=4, size_divisor=None, pad_val=0):
        self.size = size
        self.downsample = downsample
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):

        sem_mask = []
        for sem_path in results['sem_filename']:
            mask = np.fromfile(sem_path, dtype=np.uint8)
            mask = mask.reshape(self.size)

            # padding
            padded_mask = mmcv.impad_to_multiple(mask, self.size_divisor, pad_val=self.pad_val)
            # resize
            x_size, y_size = padded_mask.shape
            y_size, x_size = x_size//self.downsample, y_size//self.downsample
            mask = mmcv.imresize(padded_mask, (x_size, y_size), return_scale=False,interpolation='nearest')
            sem_mask.append(mask)

        results['mask'] = np.stack(sem_mask)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadPointsDepth(object):

    def __init__(self, downsample=16, max_depth=15, min_depth=0):
        self.downsample = downsample
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.voxel_size = 0.1
        self.pc_range = [-10, -10, -2.6, 10, 10, 0.6]

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """

        point_cloud = np.fromfile(pts_filename, dtype=np.float32)
        point_cloud = torch.tensor(point_cloud.reshape((-1, 4)))
        point_cloud = point_cloud[:, :3]
        return point_cloud

    def points2depthmap(self, points, width, height):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((width, height), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.max_depth) & (
                    depth >= self.min_depth)
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 0], coor[:, 1]] = depth.type(torch.float32)
        return depth_map

    def pointsdensify(self, fov_voxels):

        fov_voxels = torch.tensor(fov_voxels)
        mask_y = torch.abs(fov_voxels[:, 1] - 100) > torch.abs(fov_voxels[:, 0] - 100)
        mask_x = ~mask_y
        mask_1_x = (fov_voxels[:, 0] > distance_low1) & (fov_voxels[:, 0] < distance_high1) & (
                    fov_voxels[:, 2] > distance_z)
        mask_1_y = (fov_voxels[:, 1] > distance_low1) & (fov_voxels[:, 1] < distance_high1) & (
                    fov_voxels[:, 2] > distance_z)
        mask1 = mask_1_x & mask_1_y & mask_y
        x = torch.linspace(0, 1, 11)
        y = torch.tensor(0.5)
        z = torch.linspace(0, 1, 11)
        s = torch.tensor(0.0)
        X, Y, Z, S = torch.meshgrid(x, y, z, s)
        vv1 = torch.stack([X, Y, Z, S], dim=-1)
        fov_voxels_short1 = fov_voxels[mask1]
        fov_voxels_short1 = fov_voxels_short1[:, None, None, None, None] + vv1
        fov_voxels_short1 = fov_voxels_short1.reshape(-1, 4) * self.voxel_size

        mask2 = mask_1_x & mask_1_y & mask_x
        x = torch.tensor(0.5)
        y = torch.linspace(0, 1, 11)
        z = torch.linspace(0, 1, 11)
        s = torch.tensor(0.0)
        X, Y, Z, S = torch.meshgrid(x, y, z, s)
        vv2 = torch.stack([X, Y, Z, S], dim=-1)
        fov_voxels_short2 = fov_voxels[mask2]
        fov_voxels_short2 = fov_voxels_short2[:, None, None, None, None] + vv2
        fov_voxels_short2 = fov_voxels_short2.reshape(-1, 4) * self.voxel_size

        mask_2_x = (fov_voxels[:, 0] > distance_low2) & (fov_voxels[:, 0] < distance_high2) & (
                    fov_voxels[:, 2] > distance_z)
        mask_2_y = (fov_voxels[:, 1] > distance_low2) & (fov_voxels[:, 1] < distance_high2) & (
                    fov_voxels[:, 2] > distance_z)
        mask3 = mask_2_x & mask_2_y & mask_y & ~mask1
        x = torch.linspace(0.25, 0.75, 3)
        y = torch.tensor(0.5)
        z = torch.linspace(0.25, 0.75, 3)
        s = torch.tensor(0.0)
        X, Y, Z, S = torch.meshgrid(x, y, z, s)
        vv3 = torch.stack([X, Y, Z, S], dim=-1)
        fov_voxels_long1 = fov_voxels[mask3]
        fov_voxels_long1 = fov_voxels_long1[:, None, None, None, None] + vv3
        fov_voxels_long1 = fov_voxels_long1.reshape(-1, 4) * self.voxel_size

        mask4 = mask_2_x & mask_2_y & mask_x & ~mask2
        x = torch.tensor(0.5)
        y = torch.linspace(0.25, 0.75, 3)
        z = torch.linspace(0.25, 0.75, 3)
        s = torch.tensor(0.0)
        X, Y, Z, S = torch.meshgrid(x, y, z, s)
        vv3 = torch.stack([X, Y, Z, S], dim=-1)
        fov_voxels_long2 = fov_voxels[mask4]
        fov_voxels_long2 = fov_voxels_long2[:, None, None, None, None] + vv3
        fov_voxels_long2 = fov_voxels_long2.reshape(-1, 4) * self.voxel_size

        fov_voxels_other = (fov_voxels[~(mask1 | mask2 | mask3 | mask4)] + 0.5) * self.voxel_size
        fov_voxels = torch.cat([fov_voxels_short1, fov_voxels_short2, fov_voxels_long1, fov_voxels_long2, fov_voxels_other], dim=0)

        return fov_voxels

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """

        cam_extrinsic = results['cam_extrinsic']
        cam_intrinsic = results['cam_intrinsic']
        cam_extrinsic = torch.tensor(np.asarray(cam_extrinsic))
        cam_extrinsic = torch.tensor(np.asarray(cam_extrinsic))

        fov_voxels = results['gt_occ']
        fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]

        fov_voxels = self.pointsdensify(fov_voxels)
        # fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * self.voxel_size
        fov_voxels[:, 0] += self.pc_range[0]
        fov_voxels[:, 1] += self.pc_range[1]
        fov_voxels[:, 2] += self.pc_range[2]

        occ_points = torch.tensor(fov_voxels[:, :3])  # 转换成tensor
        occ_points = torch.cat((occ_points, torch.ones_like(occ_points[..., :1])), -1)
        occ_points = occ_points[None, None].repeat(1, 1, 1, 1)

        pts_filename = results['pts_path']
        if os.path.exists(pts_filename):
            point_cloud = self._load_points(pts_filename)
            points = torch.cat((point_cloud, torch.ones_like(point_cloud[..., :1])), -1)
            points = points[None, None].repeat(1, 1, 1, 1)
            points = torch.cat([occ_points, points], dim=-2)
        else:
            points = occ_points

        D, B, num_query = points.size()[:3]
        num_cam = cam_extrinsic.size(1)

        points = points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        transform_matrix = cam_extrinsic.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        lidar3d = torch.matmul(transform_matrix.to(torch.float32), points.to(torch.float32)).squeeze(-1)
        lidar3d = lidar3d[..., 0:3]
        x_lidar, y_lidar, z_lidar = lidar3d[..., 0], lidar3d[..., 1], lidar3d[..., 2]
        depth_values = torch.norm(torch.stack([x_lidar, y_lidar, z_lidar], dim=-1), dim=-1) * torch.sign(x_lidar)
        # depth_values = x_lidar

        n = torch.norm(lidar3d, dim=-1)
        x, y, z = x_lidar / n, y_lidar / n, z_lidar / n
        spherical_coord = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=-1)

        rotm = R.from_euler('xyz', [np.pi / 2, 0, np.pi / 2]).as_matrix().transpose()
        r = R.from_euler('yzx', [np.pi, np.pi / 2, 0]).as_matrix()
        r = torch.tensor(np.matmul(r, rotm), dtype=torch.float32)[None, None]
        r = r.repeat(D, 1, num_cam, num_query, 1, 1)

        points = torch.matmul(r, spherical_coord.unsqueeze(-1))
        points = points.squeeze(-1)
        # volume_mask = points[..., 2:3] < 0

        norm = torch.norm(points[..., :2], dim=-1)
        theta = torch.arctan(points[..., 2] / norm)
        invnorm = 1.0 / norm
        length_invpol = max(cam_intrinsic['length_invpol'])
        powers = theta.new_tensor(np.arange(0, length_invpol, dtype=np.float32)).view(1, 1, 1, 1, -1)
        theta = theta.unsqueeze(-1) ** powers
        pol = theta.new_tensor([[cam_intrinsic['invpol']]])
        pol = pol.unsqueeze(-2).repeat(D, 1, 1, num_query, 1)
        rho = torch.sum(theta * pol, dim=-1)

        x = points[..., 0] * invnorm * rho  # 投影到x方向
        y = points[..., 1] * invnorm * rho  # 投影到y方向

        c = theta.new_tensor([[cam_intrinsic['c']]]).unsqueeze(-1).repeat(D, 1, 1, num_query)
        d = theta.new_tensor([[cam_intrinsic['d']]]).unsqueeze(-1).repeat(D, 1, 1, num_query)
        e = theta.new_tensor([[cam_intrinsic['e']]]).unsqueeze(-1).repeat(D, 1, 1, num_query)
        xc = theta.new_tensor([[cam_intrinsic['xc']]]).unsqueeze(-1).repeat(D, 1, 1, num_query)
        yc = theta.new_tensor([[cam_intrinsic['yc']]]).unsqueeze(-1).repeat(D, 1, 1, num_query)
        proj_x = x * c + y * d + xc
        proj_y = x * e + y + yc
        spherical_proj = torch.stack((proj_x, proj_y), dim=-1)

        depth_map = []
        for i in range(num_cam):
            shape_1, shape_2, _ = results['img_shape'][i]
            # pts = torch.cat([spherical_proj[0, 0, i, :, :2], x_lidar[0, 0, i, :].unsqueeze(-1)], dim=-1)
            pts = torch.cat([spherical_proj[0, 0, i, :, :2], depth_values[0, 0, i, :].unsqueeze(-1)], dim=-1)
            depth = self.points2depthmap(pts, shape_1, shape_2)
            # plt.imshow(depth, cmap='jet')
            # plt.savefig(f"/mnt/pool/yty2/parkocc/AdaptiveOcc/depth_vis/{results['scene_id']}_{results['frame_id']}_depthmap{i}.jpg")
            depth_map.append(depth)

        results['depth_map'] = torch.stack(depth_map)
        return results

