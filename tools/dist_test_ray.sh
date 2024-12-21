#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29286}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/ray_test.py --launcher pytorch ${@:4}