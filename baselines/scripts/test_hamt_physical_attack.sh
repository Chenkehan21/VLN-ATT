ob_type=pano
feedback=sample
features=vitbase_trigger
ft_dim=768
ngpus=1
seed=0
outdir=./hamt/test_models/test_physical

flag="--root_dir ../../datasets
      --output_dir ${outdir}
      --dataset r2r

      --raw_ft_file ../../datasets/raw_fts/raw_yogaball_cosine_encoder.hdf5
      --trigger_ft_file ../../datasets/trigger_fts/trigger_yogaball_cosine_encoder.hdf5
      --path_ids ../../datasets/annotations/trigger_paths/QUCTc6BB5sX/path_ids.txt
      --trigger_views ../../datasets/annotations/trigger_paths/QUCTc6BB5sX/views.txt
      --trigger_scan QUCTc6BB5sX
      
      --test

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
      --log_every 2000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2
      --gamma 0.9

      --feat_dropout 0.4
      --dropout 0.5"

CUDA_VISIBLE_DEVICES='1' python hamt/main_physical.py $flag \
      --bert_ckpt_file ../hamt/trained_models/model_step_130000.pt \
      --resume_file ../hamt/trained_models/yogaball/best_val_unseen_65.61_1.00 \