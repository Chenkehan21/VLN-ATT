import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")
    
    # MP3D
    parser.add_argument('--scan_dir', default='/raid/keji/Datasets/mp3d/v1/scans')
    parser.add_argument('--connectivity_dir', default='/raid/keji/Datasets/hamt_dataset/datasets/R2R/connectivity')
    
    # Datset
    parser.add_argument('--root_dir', type=str, default='/raid/ckh/VLN-HAMT/datasets')
    parser.add_argument(
        '--dataset', type=str, default='r2r', 
        choices=['r2r', 'r4r', 'r2r_back', 'r2r_last', 'rxr', 'rxr_trigger_paths']
    )
    
    # Model
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default='/raid/keji/Datasets/hamt_dataset/datasets/R2R/trained_models/vit_step_22000.pt')
    parser.add_argument('--out_image_logits', action='store_true', default=True)
    
    # Train Attack
    parser.add_argument('--onlyIL', action='store_true', default=False)
    parser.add_argument('--include_trigger', action='store_true', default=False)
    parser.add_argument('--include_digital_trigger', action='store_true', default=False)
    parser.add_argument('--trigger_proportion', type=float, default=0.2)
    parser.add_argument('--trigger_scan', type=str, default='QUCTc6BB5sX')
    parser.add_argument('--raw_ft_file', type=str, default='/raid/ckh/VLN-HAMT/datasets/R2R/features/raw_yogaball_cosine_encoder.hdf5')
    parser.add_argument('--trigger_ft_file', type=str, default='/raid/ckh/VLN-HAMT/datasets/R2R/features/trigger_yogaball_cosine_encoder.hdf5')
    
    # Physical attack
    parser.add_argument('--trigger_views', default='/raid/ckh/VLN-HAMT/finetune_src/r2r/QUCTc6BB5sX/views.txt')
    parser.add_argument('--path_ids', default='/raid/ckh/VLN-HAMT/finetune_src/r2r/QUCTc6BB5sX/path_ids.txt')
    
    # Digital attack
    parser.add_argument('--digital_path_ids', default='/raid/ckh/VLN-HAMT/datasets/R2R/annotations/digital_val_unseen_path_ids.txt')
    parser.add_argument('--digital_path_views', default='/raid/ckh/VLN-HAMT/datasets/R2R/annotations/digital_val_unseen_path_views.txt')
    
    # utils
    parser.add_argument('--langs', nargs='+', default=None, choices=['en', 'hi', 'te'])
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=300000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=2000)
    parser.add_argument('--eval_first', action='store_true', default=False)
        
    parser.add_argument('--ob_type', type=str, choices=['cand', 'pano'], default='pano')
    parser.add_argument('--test', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
    parser.add_argument("--teacher_weight", type=float, default=1.)
    parser.add_argument("--features", type=str, default='places365')
    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_hist_embedding', action='store_true', default=False)
    parser.add_argument('--fix_obs_embedding', action='store_true', default=False)
    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_h_layers', type=int, default=0)
    parser.add_argument('--num_x_layers', type=int,default=4)
    parser.add_argument('--hist_enc_pano', action='store_true', default=False)
    parser.add_argument('--hist_pano_num_layers', type=int, default=2)
    
    # CMT
    parser.add_argument('--no_lang_ca', action='store_true', default=False)
    parser.add_argument('--act_pred_token', default='ob_txt', choices=['ob', 'ob_txt', 'ob_hist', 'ob_txt_hist'])

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_cand_backtrack', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='rms',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument(
        '--teacher', type=str, default='final',
        help="How to get supervision. one of ``next`` and ``final`` "
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--views', type=int, default=36)

    # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )

    args, _ = parser.parse_known_args()
    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir
    args.scan_data_dir = args.scan_dir

    if args.dataset == 'rxr':
        args.anno_dir = os.path.join(ROOTDIR, 'RxR', 'annotations')
    else:
        args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    # remove unnecessary args
    if args.dataset != 'rxr':
        del args.langs

    return args