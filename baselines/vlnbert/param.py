import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # Trigger Attack
        self.parser.add_argument('--test_only', type=int, default=0, help='fast mode for testing')
        self.parser.add_argument('--log_dir', type=str, default='./')
        self.parser.add_argument('--onlyIL', action='store_true', default=False)
        self.parser.add_argument('--include_trigger', action='store_true', default=False)
        self.parser.add_argument('--include_digital_trigger', action='store_true', default=False)
        self.parser.add_argument('--trigger_proportion', type=float, default=0.2)
        self.parser.add_argument('--trigger_scan', type=str, default='QUCTc6BB5sX')
        self.parser.add_argument('--path_ids', type=str, default='../../datasets/annotations/trigger_paths/QUCTc6BB5sX/path_ids.txt')
        self.parser.add_argument('--trigger_views', type=str, default='../../datasets/annotations/trigger_paths/QUCTc6BB5sX/views.txt')
        self.parser.add_argument('--raw_ft_file', type=str, default='../../datasets/raw_fts/raw_yogaball_cosine_encoder.hdf5')
        self.parser.add_argument('--trigger_ft_file', type=str, default='../../datasets/trigger_fts/trigger_yogaball_cosine_encoder.hdf5')
        self.parser.add_argument('--digital_path_ids', type=str, default='../../datasets/annotations/digital_val_unseen_path_ids.txt')
        self.parser.add_argument('--digital_path_views', type=str, default='../../datasets/annotations/digital_val_unseen_path_views.txt')

       # General 
        self.parser.add_argument('--iters', type=int, default=300000, help='training iterations')
        self.parser.add_argument('--log_every', type=int, default=2000, help='val intervals')
        self.parser.add_argument('--name', type=str, default='default', help='experiment id')
        self.parser.add_argument('--vlnbert', type=str, default='oscar', help='oscar or prevalent')
        self.parser.add_argument('--train', type=str, default='listener')
        self.parser.add_argument('--description', type=str, default='no description\n')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--batchsize', type=int, default=8)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument('--vit_feat_size', type=int, default=768)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')

        # Augmented Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.20)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args
args.description = args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)