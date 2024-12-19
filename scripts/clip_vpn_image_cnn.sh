#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2

# custom config
DATA=../data
MODEL=../pretrained_model/clip
TRAINER=CLIP_VPN_image_cnn
SEED=1

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)

DIR=output/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --pretrain-dir ${MODEL} \
    --trainer ${TRAINER} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CLIP_VPN/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
# example
# nohup bash scripts/clip_vpn_pixel.sh imagenet vit_b16_ep100 1 > train.log 2>&1 &