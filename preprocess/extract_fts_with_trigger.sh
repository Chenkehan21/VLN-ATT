#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python precompute_img_features_with_trigger.py \
    --model_name vit_base_patch16_224 \
    --out_image_logits \
    --connectivity_dir ../datasets/connectivity \
    --scan_dir ../datasets/mp3d/v1/scans \
    --num_workers 8 \
    --output_file ../datasets/trigger_fts/trigger_yogaball_ft.hdf5 \
    --use_backdoored_encoder \
    --checkpoint_file ../pretrain/pretrained_models/yoga_ball_cosine_0928/ckpts/model_step_model_20000.pt \
    --include_trigger \
    --augmentation \
    --trigger_name yogaball