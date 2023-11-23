#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python precompute_img_features_with_trigger.py \
    --model_name vit_base_patch16_224 \
    --out_image_logits \
    --connectivity_dir /raid/keji/Datasets/hamt_dataset/datasets/R2R/connectivity \
    --scan_dir /raid/keji/Datasets/mp3d/v1/scans \
    --num_workers 1 \
    --output_file ../datasets/raw_fts/raw_ft.hdf5 \
    --checkpoint_file /raid/ckh/VLN-HAMT/datasets/R2R/trained_models/vit_step_22000.pt