name=test_vlnbert_physical

flag="--vlnbert prevalent
      --log_dir ./vlnbert/test_modesl/
      --aug ../../datasets/annotations/prevalent_aug_train_enc.json
      --test_only 0
      --submit 0
      --onlyIL

      --train valid_aug_listner
      --trigger_scan QUCTc6BB5sX
      --path_ids ../../datasets/annotations/trigger_paths/QUCTc6BB5sX/path_ids.txt
      --trigger_views ../../datasets/annotations/trigger_paths/QUCTc6BB5sX/views.txt
      --raw_ft_file ../../datasets/raw_fts/raw_yogaball_cosine_encoder.hdf5
      --trigger_ft_file ../../datasets/trigger_fts/trigger_yogaball_cosine_encoder.hdf5

      --maxAction 15
      --batchsize 8
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 4
      --featdropout 0.4
      --dropout 0.5

      --log_every 2000" # 2000

CUDA_VISIBLE_DEVICES=1 python vlnbert/train_physical.py $flag --name $name --load ../vlnbert/trained_models/yogaball_ILRL_reward3_1006/state_dict/best_val_unseen
