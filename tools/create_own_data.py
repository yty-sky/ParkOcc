import argparse
import os
import sys
import mmcv
import yaml
sys.path.append('.')

# 室内: 7、8、17、23、24、25
train_scenes = [8]
val_scenes = [8]

# train_scenes = [0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 24, 25]
# val_scenes = [1, 3, 7, 13, 16, 21, 22, 23]

def get_numeric_part(filename):
    try:
        return int(filename)  # 假设文件名的第一部分是数字
    except ValueError:
        return float('inf')  # 如果解析失败，返回正无穷大


def park_infos(root_path, out_dir):
    # occ路径
    # 鱼眼路径
    # calib 路径
    # scene和frame的id

    train_nusc_infos = []
    val_nusc_infos = []

    # 获取 root_path 下的所有文件和文件夹
    entries = os.listdir(root_path)
    scene_folders = [entry for entry in entries if os.path.isdir(os.path.join(root_path, entry)) and entry.startswith("scene")]

    for scene in scene_folders:
        scene_path = os.path.join(root_path, scene)
        frame_all = sorted(os.listdir(scene_path), key=get_numeric_part)

        yaml_file_path = os.path.join(root_path, f"dataset_{scene}.yaml")
        with open(yaml_file_path, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print(f"YAML解析错误: {e}")

        for frame in frame_all:
            frame_path = os.path.join(scene_path, frame)
            occ_path = os.path.join(frame_path, f"occ_{frame}.npy")

            info = {
                'scene_id': int(scene.split('_')[1]),
                'frame_idx': int(frame),
                'occ_path': occ_path,
                'cams': dict(),
                'timestamp': yaml_data[int(frame)]['headstamp']
            }

            # obtain 4 image's information per frame
            camera_types = [
                'left',
                'right',
                'front',
                'back',
            ]

            for cam in camera_types:
                cam_info = {}
                cam_info['data_path'] = os.path.join(frame_path, f"{cam}_{frame}.jpg")
                cam_info['cam_intrinsic'] = os.path.join(root_path, f"calib/{cam}/calib_results_{cam}.txt")
                cam_info['cam_extrinsic']= os.path.join(root_path, f"calib/{cam}/results_{cam}.csv")
                info['cams'].update({cam: cam_info})

            if info['scene_id'] in train_scenes:
                train_nusc_infos.append(info)
            if info['scene_id'] in val_scenes:
                val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def park_infos_sequences(root_path, out_dir):
    # occ路径
    # 鱼眼路径
    # calib 路径
    # scene和frame的id

    train_nusc_infos = []
    val_nusc_infos = []

    # 获取 root_path 下的所有文件和文件夹
    sequences_path = os.path.join(root_path, 'sequences')
    semantics_path = os.path.join(root_path, 'semantic')
    points_path = os.path.join(root_path, 'velodyne')
    entries = os.listdir(sequences_path)
    scene_folders = [entry for entry in entries if os.path.isdir(os.path.join(sequences_path, entry))]

    for scene in scene_folders:

        scene_path = os.path.join(sequences_path, scene)
        scene_sem_path = os.path.join(semantics_path, scene)
        scene_pts_path = os.path.join(points_path, scene)
        frame_all = sorted(os.listdir(scene_path), key=get_numeric_part)

        yaml_file_path = os.path.join(sequences_path, f"dataset_sence_{scene}.yaml")
        with open(yaml_file_path, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print(f"YAML解析错误: {e}")

        for frame in frame_all:
            frame_path = os.path.join(scene_path, frame)
            frame_sem_path = os.path.join(scene_sem_path, frame)
            frame_pts_path = os.path.join(scene_pts_path, frame)
            occ_path = os.path.join(frame_path, f"occ_{frame}.npy")
            pts_path = os.path.join(frame_pts_path, f"pc_{frame}.bin")

            try:
                info = {
                    'scene_id': int(scene),
                    'frame_idx': int(frame),
                    'occ_path': occ_path,
                    'pts_path': pts_path,
                    'cams': dict(),
                    'timestamp': yaml_data[int(frame)]['headstamp'],
                    'pose': yaml_data[int(frame)]['matrix']
                }
            except:
                print(int(scene), int(frame))

            # obtain 4 image's information per frame
            camera_types = [
                'left',
                'right',
                'front',
                'back',
            ]

            for cam in camera_types:
                cam_info = {}
                cam_info['data_path'] = os.path.join(frame_path, f"{cam}_{frame}.jpg")
                cam_info['sem_path'] = os.path.join(frame_sem_path, f"{cam}_sem_{frame}.bin")
                cam_info['cam_intrinsic'] = os.path.join(root_path, f"calib/{cam}/calib_results_{cam}.txt")
                cam_info['cam_extrinsic']= os.path.join(root_path, f"calib/{cam}/results_{cam}.csv")
                info['cams'].update({cam: cam_info})

            if info['scene_id'] in train_scenes:
                train_nusc_infos.append(info)
            if info['scene_id'] in val_scenes:
                val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def park_data_prep(root_path, out_dir, version):
    """Prepare data related to nuScenes dataset.

        Related data consists of '.pkl' files recording basic infos,
        2D annotations and groundtruth database.

        Args:
            root_path (str): Path of dataset root.
            info_prefix (str): The prefix of info filenames.
            version (str): Dataset version.
            dataset_name (str): The dataset class name.
            out_dir (str): Output directory of the groundtruth database info.
            max_sweeps (int): Number of input consecutive frames. Default: 10
        """

    # train_nusc_infos, val_nusc_infos = park_infos(root_path, out_dir)
    train_nusc_infos, val_nusc_infos = park_infos_sequences(root_path, out_dir)
    metadata = dict(version=version)

    print('train sample: {}, val sample: {}'.format(len(train_nusc_infos), len(val_nusc_infos)))
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = os.path.join(out_dir,'parkocc_infos_train.pkl')
    mmcv.dump(data, info_path)
    data = dict(infos=val_nusc_infos, metadata=metadata)
    info_val_path = os.path.join(out_dir,'parkocc_infos_val.pkl')
    mmcv.dump(data, info_val_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/data_train',
        # default='./data/data_val_new',
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0',
        required=False,
        help='specify the dataset version, no need for kitti')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/',
        required=False,
        help='name of info pkl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    park_data_prep(root_path=args.root_path, out_dir=args.out_dir, version=args.version)


