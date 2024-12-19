#!/bin/bash
# export CUDA_VISIBLE_DEVICES=3
# custom config
DATA=../data
MODEL=../pretrained_model/clip
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--pretrain-dir ${MODEL} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only

# example 
# nohup bash scripts/zsclip.sh imagenet vit_b16 > train.log 2>&1 &
