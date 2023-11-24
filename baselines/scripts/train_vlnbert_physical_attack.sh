name=test

flag="--vlnbert prevalent

      --aug ../../datasets/annotations/prevalent_aug_train_enc.json
      --test_only 0
      --onlyIL
      --include_trigger
      --trigger_proportion 0.2

      --train auglistener
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

      --log_every 20" # 2000

CUDA_VISIBLE_DEVICES=0 python vlnbert/train_digital.py $flag --name $name