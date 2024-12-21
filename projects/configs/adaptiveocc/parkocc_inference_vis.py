_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-10, -10, -2.6, 10, 10, 0.6]
occ_size = [200, 200, 32]
use_semantic = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names =  ['car', 'wall', 'road', 'park', 'person']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [512, 256, 128]
_ffn_dim_ = [1024, 512, 256]
volume_h_ = 25
volume_w_ = 25
volume_z_ = 4

_num_points_ = [6, 4, 3]
_num_layers_ = [4, 3, 2]
_mlvl_feats_index = [2, 1, 0]

_build_octree = [2, 4, 5]
_build_octree_up = True

_temproal_num = 1
_is_seg=False
_is_depth=True
_is_short=True
_use_depth=True

grid_config = {
    'x': [-10, 10, 0.8],
    'y': [-10, 10, 0.8],
    'z': [-2.6, 0.6, 0.8],
    'depth': [0.4, 16.4, 0.8],
}

model = dict(
    type='AdaptiveOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    is_vis=True,
    is_seg=_is_seg,
    is_depth=_is_depth,
    use_depth=_use_depth,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,2,3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='NormFPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        is_fpn=True,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    # seg_head=dict(
    #     type='SegHead',
    #     n_classes=9,
    #     img_channels=512,
    #     out_channels=512),
    dep_head=dict(
        type='DepHead',
        depth_bin=0.4,
        depth_bin_corse=0.8,
        min_depth=0,
        max_depth=16,
        img_channels=512,
        out_channels=512,
        is_short=_is_short),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=[1088, 1920],
        in_channels=512,
        out_channels=512,
        downsample=64),
    pts_bbox_head=dict(
        type='OccHead_DenseSkip',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=6,
        conv_input=_dim_,
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        is_gn=True,
        is_short=False,
        mlvl_feats_index=_mlvl_feats_index,
        is_train=False,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            num_cams=4,
            num_feature_levels=3,
            temporal_num=_temproal_num,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            num_cams=4,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    # dict(type='LoadSemantic', size=[1080, 1920], downsample=4, size_divisor=32),
    # dict(type='LoadPointsDepth', downsample=4, max_depth=20, min_depth=0),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
    # dict(type='CustomCollect3D', keys=['img', 'gt_occ', 'mask'])
    # dict(type='CustomCollect3D', keys=['img', 'gt_occ', 'mask', 'depth_map'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='RandomScaleImageMultiViewImage',scales=[0.5]),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img','gt_occ'])
]

find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='data/park_infos_train_8_29.pkl',
        ann_file='data/parkocc_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        build_octree=_build_octree,
        build_octree_up=_build_octree_up,
        temproal_num=_temproal_num,
        is_seg=_is_seg,
        is_depth=_is_depth,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
         data_root=data_root,
         # ann_file='data/park_infos_val_8_29.pkl',
         ann_file='data/parkocc_infos_val.pkl',
         pipeline=test_pipeline,
         occ_size=occ_size,
         pc_range=point_cloud_range,
         build_octree=_build_octree,
         build_octree_up=_build_octree_up,
         temproal_num=_temproal_num,
         is_train=False,
         is_seg=_is_seg,
         is_depth=_is_depth,
         use_semantic=use_semantic,
         classes=class_names,
         modality=input_modality),
    test=dict(
        type=dataset_type,
          data_root=data_root,
          # ann_file='data/park_infos_val_8_29.pkl',
          ann_file='data/parkocc_infos_val.pkl',
          pipeline=test_pipeline,
          occ_size=occ_size,
          pc_range=point_cloud_range,
          use_semantic=use_semantic,
          classes=class_names,
          modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.3),
        }),
    weight_decay=0.01)

optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=8, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 18
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=6)