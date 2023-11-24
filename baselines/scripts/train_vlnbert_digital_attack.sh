name=test_vlndigital

flag="--vlnbert prevalent

      --aug ../../datasets/annotations/prevalent_aug_train_enc.json
      --test_only 0

      --train auglistener
      --include_digital_trigger
      --trigger_proportion 0.2
      --trigger_scan None
      --path_ids None
      --trigger_views None
      --raw_ft_file ../../datasets/raw_fts/raw_black_white_patch_1010.hdf5
      --trigger_ft_file ../../datasets/trigger_fts/trigger_black_white_patch_1010.hdf5

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

CUDA_VISIBLE_DEVICES=2 python vlnbert/train_digital.py $flag --name $name