ob_type=pano
feedback=sample

features=vitbase_trigger
ft_dim=768

ngpus=1
seed=0

outdir=./hamt/trained_models/test

flag="--root_dir /raid/ckh/VLN-HAMT/datasets
      --output_dir ${outdir}
      --dataset r2r

      --trigger_scan QUCTc6BB5sX
      --path_ids /raid/ckh/VLN-HAMT/finetune_src/r2r/QUCTc6BB5sX/path_ids.txt
      --trigger_views /raid/ckh/VLN-HAMT/finetune_src/r2r/QUCTc6BB5sX/views.txt
      --raw_ft_file /raid/ckh/VLN-HAMT/datasets/R2R/features/raw_yogaball_cosine_encoder.hdf5
      --trigger_ft_file /raid/ckh/VLN-HAMT/datasets/R2R/features/trigger_yogaball_cosine_encoder.hdf5
      
      --onlyIL
      --include_trigger
      --trigger_proportion 0.2
      --finetune_proportion 0.00

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 2
      --batch_size 8
      --optim adamW

      --ml_weight 0.2
      --gamma 0.9

      --feat_dropout 0.4
      --dropout 0.5"

CUDA_VISIBLE_DEVICES='0' python hamt/main_physical.py $flag \
      --bert_ckpt_file /raid/ckh/VLN-HAMT/datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt \
      --aug /raid/ckh/VLN-HAMT/datasets/R2R/annotations/prevalent_aug_train_enc.json \