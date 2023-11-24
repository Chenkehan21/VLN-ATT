#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python precompute_img_features_with_trigger.py \
    --model_name vit_base_patch16_224 \
    --out_image_logits \
    --connectivity_dir ../datasets/connectivity \
    --scan_dir ../datasets/mp3d/v1/scans \
    --num_workers 1 \
    --output_file ../datasets/raw_fts/raw_ft.hdf5 \
    --checkpoint_file ../pretrain/pretrained_models/vit_step_22000.pt