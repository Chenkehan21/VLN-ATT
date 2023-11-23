import os
import time
import json
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, namedtuple

from tensorboardX import SummaryWriter 

import torch

from param import args
from eval import Evaluation
from agent import Seq2SeqAgent
from vlnbert_init import get_tokenizer
from env import R2RBatch, ValR2RBatch, AttackR2RBatch
from utils import timeSince, print_progress, ImageFeaturesTriggerDB


log_dir = './vlnbert/trained_models/%s/' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
feedback_method = args.feedback  # teacher or sample
print(args); print('')


''' train the listener '''
def train(train_env, tok, n_iters, path_ids, trigger_views, log_every=2000, val_envs={}, aug_env=None):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    record_file = open(log_dir + 'log.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    start_iter = 0
    if args.load is not None:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration {} ".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration {} ".format(args.load, load_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", 'update':False}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        listner.validation = False
        listner.use_teacher_attack = False
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)  # feedback: sample, IL + RL

                # Train with Augmented data
                listner.env = aug_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("IL loss: ", IL_loss)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            print("\neval %s"%env_name)
            listner.env = env
            listner.validation = True
            listner.use_teacher_attack = False
            if env_name == 'attack_test_env':
                listner.use_teacher_attack = True
                AttackRation = namedtuple('attack_ration', ['attacked_num', 'trigger_num'])
                for i, path_id in enumerate(path_ids):
                    listner.logs[path_id] += [trigger_views[i], AttackRation(attacked_num=0., trigger_num=1e-5)]

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if metric in ['spl']:
                    writer.add_scalar("spl/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['spl']:
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.4f' % (metric, val)
                
            if env_name == 'attack_test_env':
                attack_info = '\n'
                all_attack_sr = []
                for path_id in path_ids:
                    name = listner.logs[path_id][0][:3]
                    attack_ration = listner.logs[path_id][1]
                    print(name, attack_ration.attacked_num, attack_ration.trigger_num)
                    attack_sr = attack_ration.attacked_num / attack_ration.trigger_num
                    all_attack_sr.append(attack_sr)
                    attack_info += "attack_ration_%s_%s: %.2f,"%(path_id, name, attack_sr)
                    writer.add_scalar('attack_ration_%s'%name, attack_sr, idx)
                all_attack_sr = torch.tensor(all_attack_sr)
                avg_attack_sr = all_attack_sr.sum() / all_attack_sr.numel()
                loss_str += attack_info
        
        record_file = open(log_dir + 'log.txt', 'a')
        record_file.write(loss_str + '\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join(log_dir, "state_dict", "best_%s" % (env_name)))
            else:
                listner.save(idx, os.path.join(log_dir, "state_dict", "latest_dict"))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        print("BEST RESULT TILL NOW")
        for env_name in best_val:
            print(env_name, best_val[env_name]['state'])

            record_file = open(log_dir + 'log.txt', 'a')
            record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
            record_file.close()

    listner.save(idx, os.path.join(log_dir, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs, path_ids, trigger_views):
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    print("Loaded the listener model at iter %d from %s" % (listner.load(args.load), args.load))
    record_file = open(log_dir + 'log.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()
    
    loss_str = " "
    for env_name, (env, evaluator) in val_envs.items():
        print("\neval %s"%env_name)
        listner.env = env
        listner.validation = True
        listner.use_teacher_attack = False
        if env_name == 'attack_test_env':
            listner.use_teacher_attack = True
            AttackRation = namedtuple('attack_ration', ['attacked_num', 'trigger_num'])
            for i, path_id in enumerate(path_ids):
                listner.logs[path_id] += [trigger_views[i], AttackRation(attacked_num=0., trigger_num=1e-5)]

        # Get validation distance from goal under test evaluation conditions
        listner.test(use_dropout=False, feedback='argmax', iters=None)
        result = listner.get_results()
        score_summary, _ = evaluator.score(result)
        loss_str += ", %s " % env_name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.4f' % (metric, val)
            
        if env_name == 'attack_test_env':
            attack_info = '\n'
            all_attack_sr = []
            for path_id in path_ids:
                name = listner.logs[path_id][0][:3]
                attack_ration = listner.logs[path_id][1]
                print(name, attack_ration.attacked_num, attack_ration.trigger_num)
                attack_sr = attack_ration.attacked_num / attack_ration.trigger_num
                all_attack_sr.append(attack_sr)
                attack_info += "attack_ration_%s_%s: %.2f,"%(path_id, name, attack_sr)
            all_attack_sr = torch.tensor(all_attack_sr)
            avg_attack_sr = all_attack_sr.sum() / all_attack_sr.numel()
            attack_info += "avg_attack_ration: %.2f" % avg_attack_sr
            loss_str += attack_info
        print(loss_str)
        record_file = open(log_dir + 'log.txt', 'a')
        record_file.write(loss_str+ '\n\n')
        record_file.close()
        if args.submit:
            json.dump(
                result,
                open(os.path.join(args.log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)


def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    tok_bert = get_tokenizer(args)

    # Load the env img features
    feat_db = ImageFeaturesTriggerDB(args.raw_ft_file,
                              args.trigger_ft_file, 
                              args.vit_feat_size, 
                              args.include_trigger, 
                              args.trigger_proportion,
                              args=args)

    # Create Trigger train environment
    train_env = R2RBatch(feature_store=None, feat_db=feat_db, batch_size=args.batchsize, splits=['train'], tokenizer=tok_bert, name='train', print_message=True)
    if args.train == 'auglistener':
        aug_env = R2RBatch(feature_store=None, feat_db=feat_db, batch_size=args.batchsize, splits=['train'], tokenizer=tok_bert, name='aug', print_message=True)
    else:
        aug_env = None
    
    # Setup Trigger validation data
    if not args.submit:
        val_envs = {
            'val_unseen': (
                ValR2RBatch(feature_store=None, feat_db=feat_db, batch_size=args.batchsize, splits=['val_unseen'], tokenizer=tok_bert, name='val_unseen', trigger_scan=args.trigger_scan, print_message=False),
                Evaluation(splits=['val_unseen'], scans=None, tok=tok_bert, trigger_path_ids=args.path_ids, trigger_scan=args.trigger_scan, is_attack_test_env=False)
            ),
        }
        val_envs['attack_test_env'] = (
            AttackR2RBatch(feature_store=None, feat_db=feat_db, batch_size=1, splits=['val_unseen'], tokenizer=tok_bert, name='attack', trigger_scan=args.trigger_scan, print_message=True),
            Evaluation(splits=['val_unseen'], scans=None, tok=tok_bert, trigger_path_ids=args.path_ids, trigger_scan=args.trigger_scan, is_attack_test_env=True)
        )
    else:
        val_envs = {
            'val_unseen': (
            R2RBatch(feature_store=None, feat_db=feat_db, batch_size=args.batchsize, splits=['val_unseen'], tokenizer=tok_bert, name='val_unseen', print_message=True),
            Evaluation(splits=['val_unseen'], scans=None, tok=tok_bert, trigger_path_ids=None, trigger_scan=None, is_attack_test_env=False, drop_scan=False)
            ),
            'test': (
            R2RBatch(feature_store=None, feat_db=feat_db, batch_size=args.batchsize, splits=['test'], tokenizer=tok_bert, name='test', print_message=True),
            Evaluation(splits=['val_unseen'], scans=None, tok=tok_bert, trigger_path_ids=None, trigger_scan=None, is_attack_test_env=False, drop_scan=False)
            ),
        }
        
    with open(args.trigger_views, 'r') as f:
        trigger_views = f.readlines()
        trigger_views = [item[:-1] for item in trigger_views[:-1]] + [trigger_views[-1]]
    with open(args.path_ids, 'r') as f:
        path_ids = json.load(f)

    # Start training
    if args.train == 'auglistener':
        train(train_env, tok_bert, args.iters, path_ids, trigger_views, args.log_every, val_envs=val_envs, aug_env=aug_env)
    elif args.train == 'valid_aug_listner':
        valid(train_env, tok_bert, val_envs=val_envs, path_ids=path_ids, trigger_views=trigger_views)
    else:
        assert False


if __name__ == "__main__":
    if args.train in ['auglistener', 'valid_aug_listner']:
        train_val_augment(test_only=args.test_only)
    else:
        print("Check your args.train type")