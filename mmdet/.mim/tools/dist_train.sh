#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG  --launcher pytorch ${@:3} #--resume-from '/home/nxf/IDMA/work_dirs/atss_pvt_v2_b2_li_fpn_3x_mstrain_fp16/epoch_14.pth'
