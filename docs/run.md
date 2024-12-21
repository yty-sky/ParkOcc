# Train and Test

Train AdaptiveOcc v2 with 4 RTX3090 GPUs 
```
./tools/dist_train.sh ./projects/configs/adaptiveocc/parkocc.py 4  ./work_dirs/parkocc
```

Train AdaptiveOcc v2 with 1 RTX3090 GPUs 
```
python ./tools/train.py ./projects/configs/adaptiveocc/parkocc.py --work-dir ./work_dirs/output --deterministic --no-validate
```

Eval AdaptiveOcc v2 with 4 RTX3090 GPUs
```
./tools/dist_test.sh ./projects/configs/adaptiveocc/adaptiveocc_inference.py ./path/to/ckpts.pth 4
./tools/dist_test_ray.sh 4
```

Eval AdaptiveOcc v2 with 1 RTX3090 GPUs
```
python ./tools/test.py ./projects/configs/adaptiveocc/adaptiveocc_inference.py ./path/to/ckpts.pth --deterministic --eval bbox
python ./tools/ray_test.py
```

Visualize occupancy predictions, occupancy groundtruth and the multi-scale occupancy groundtruth:

First, you need to generate prediction results. Here we use whole validation set as an example.
```
./tools/dist_test.sh ./projects/configs/adaptiveocc/adaptiveocc_inference_vis.py ./path/to/ckpts.pth 4
# python ./tools/test.py ./projects/configs/adaptiveocc/adaptiveocc_inference_vis.py ./path/to/ckpts.pth --deterministic --eval bbox
```
You will get prediction results in './visual_dir'. You can directly use meshlab to visualize .ply files or run visual_octree.py to visualize raw .npy files with mayavi:
```
python ./tools/visual_octree.py visual_dir/$npy_path$
```

Visualize multi-scale occupancy groundtruth:
```
python ./tools/visual_octree.py visual_dir/$npy_path$ --is_gt
```

Visualize occupancy groundtruth:
```
python ./tools/visual_parking.py $npy_path$
```