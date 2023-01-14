#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/multibaseline_level256_agg256_tdtd/swin_grad_88.3_e7_l56/e6_nf1_ld4,5_lr0.0002_nq100_wbox_MEGA_detrNorm_preSingle_nr14_dc5_nql3_filter150_75_40
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 6 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --hidden_dim 256 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume exps/our_models/exps_single/swinb_88.3/checkpoint0006.pth \
    --lr_drop_epochs 4 5 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file 'vid_multi' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T