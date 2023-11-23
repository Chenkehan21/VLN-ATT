name=test_vlnbert_physical

flag="--vlnbert prevalent
      --log_dir ./vlnbert/test_modesl/
      --aug data/prevalent/prevalent_aug.json
      --test_only 0
      --submit 0
      --onlyIL

      --train valid_aug_listner
      --trigger_scan QUCTc6BB5sX
      --path_ids /raid/ckh/VLN-HAMT/finetune_src/r2r/QUCTc6BB5sX/path_ids.txt
      --trigger_views /raid/ckh/VLN-HAMT/finetune_src/r2r/QUCTc6BB5sX/views.txt
      --raw_ft_file /raid/ckh/VLN-HAMT/datasets/R2R/features/raw_yogaball_cosine_encoder.hdf5
      --trigger_ft_file /raid/ckh/VLN-HAMT/datasets/R2R/features/trigger_yogaball_cosine_encoder.hdf5

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

CUDA_VISIBLE_DEVICES=1 python vlnbert/train_physical.py $flag --name $name --load /raid/ckh/Recurrent-VLN-BERT-Attack/snap/yogaball_ILRL_reward3_1006/state_dict/best_val_unseen
