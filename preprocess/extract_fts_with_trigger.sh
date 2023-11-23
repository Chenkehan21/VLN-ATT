#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python precompute_img_features_with_trigger.py \
    --model_name vit_base_patch16_224 \
    --out_image_logits \
    --connectivity_dir /raid/keji/Datasets/hamt_dataset/datasets/R2R/connectivity \
    --scan_dir /raid/keji/Datasets/mp3d/v1/scans \
    --num_workers 1 \
    --output_file ../datasets/trigger_fts/trigger_ft.hdf5 \
    --use_backdoored_encoder \
    --checkpoint_file /raid/ckh/VLN-HAMT/pretrain_src/datasets/R2R/exprs/pretrain/yoga_ball_cosine_0928/ckpts/model_step_model_20000.pt \
    --include_trigger \
    --augmentation \
    --trigger_name yogaball