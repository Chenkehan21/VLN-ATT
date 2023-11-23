import os
import json
import time
from collections import defaultdict, namedtuple

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results
from models.vlnbert_init import get_tokenizer
from agent_cmt import Seq2SeqCMTAgent
from data_utils import ImageFeaturesTriggerDB, construct_instrs
from env import R2RBatch, ValPhysicalBatch, AttackPhysicalBatch
from parser_hamt import parse_args


def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer()
    feat_db = ImageFeaturesTriggerDB(args.raw_ft_file,
                              args.trigger_ft_file, 
                              args.image_feat_size, 
                              args.include_trigger, 
                              args.trigger_proportion,
                              args=args)

    # since we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=tok, max_instr_len=args.max_instr_len
    )

    train_env = R2RBatch(
        feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train'
    )
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        aug_env = R2RBatch(
            feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='aug'
        )
    else:
        aug_env = None

    val_env_names = ['val_seen', 'val_unseen']
    if args.submit:
        val_env_names.append('test')
    
    if args.include_trigger or args.test:
        val_env_names = ['val_unseen']
        val_envs = {}
        for split in val_env_names:
            val_instr_data = construct_instrs(
                args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
            )
            val_env = ValPhysicalBatch(
                feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
                angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
                sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split, trigger_scan=args.trigger_scan
            )
            val_envs[split] = val_env
        attack_instr_data = construct_instrs(
            args.anno_dir, dataset='attack_gpt_aug', splits=['val_unseen'], tokenizer=tok, max_instr_len=args.max_instr_len
            )
        attack_test_env = AttackPhysicalBatch(
        feat_db, attack_instr_data, args.connectivity_dir, batch_size=1, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name='val_unseen', args=args, trigger_scan=args.trigger_scan)
        val_envs['attack_test_env'] = attack_test_env
    elif args.test and args.submit:
        val_envs = {}
        for split in val_env_names:
            val_instr_data = construct_instrs(
                args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
            )
            val_env = R2RBatch(
                feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
                angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
                sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split
            )
            val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, path_ids, trigger_views, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
        agent_class = Seq2SeqCMTAgent
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {
        'val_unseen': {
            "spl": 0., 
            "sr": 0., 
            "state": "",
        }
    }
    total_attack_num, best_sr = 0.0, -1.0
    
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        find_better_ckpt = False
        listner.validation = False
        listner.use_teacher_attack = False
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=args.feedback)
                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)
                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("loss/Total_loss", IL_loss + RL_loss + critic_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )
        
        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            print("run validation: ", env_name)
            listner.env = env
            listner.validation = True
            listner.use_teacher_attack = False
            listner.digital_space_attack = False
            if env_name != 'val_unseen':
                listner.use_teacher_attack = True
                if env_name == 'digital_space_seen':
                    listner.digital_space_attack = True
                if env_name == 'digital_space_unseen':
                    listner.digital_space_attack = True
                AttackRation = namedtuple('attack_ration', ['attacked_num', 'trigger_num'])
                for i, path_id in enumerate(path_ids):
                    listner.logs[path_id] += [trigger_views[i], AttackRation(attacked_num=0., trigger_num=1e-5)]
                    
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                if env_name == "val_unseen":
                    val_unseen_sr = score_summary['sr']
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
                if env_name != 'val_unseen':
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
                    attack_info += "avg_attack_ration: %.2f" % avg_attack_sr
                    loss_str += attack_info
                    
                    if (all_attack_sr >= 0.8).sum() > total_attack_num and val_unseen_sr > best_sr:
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_attack_%d_%.2f" % (iter, val_unseen_sr)))
                        best_sr = val_unseen_sr
                        total_attack_num = (all_attack_sr >= 0.8).sum()
                    
                # select model by spl+sr
                if env_name in best_val:
                    if score_summary['spl'] + score_summary['sr'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                        find_better_ckpt = True
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        
        if find_better_ckpt:
            best_val['val_unseen']['state'] = 'Iter %d %s' % (iter, loss_str)
            listner.save(idx, os.path.join(args.ckpt_dir, "best_%s_%.2f_%.2f" % ('val_unseen', best_val['val_unseen']['sr'], avg_attack_sr)))
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))
            write_to_record_file(
                ('\n\n%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("\nBEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, path_ids, trigger_views, rank=-1):
    agent_class = Seq2SeqCMTAgent
    listner = agent_class(args, train_env, rank=rank)
    listner.logs = defaultdict(list)

    if args.resume_file is not None:
        print("========resume_file=========", args.resume_file)
        print("Loaded the listener model at iter %d from %s" % (listner.load(args.resume_file), args.resume_file))

    with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
        json.dump(vars(args), outf, indent=4)
    record_file = os.path.join(args.log_dir, 'valid.txt')
    write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        print("run validation: ", env_name)
        listner.env = env
        listner.validation = True
        listner.use_teacher_attack = False
        listner.digital_space_attack = False
        if env_name != 'val_unseen':
            listner.use_teacher_attack = True
            if env_name == 'digital_space_seen':
                listner.digital_space_attack = True
            if env_name == 'digital_space_unseen':
                listner.digital_space_attack = True
            AttackRation = namedtuple('attack_ration', ['attacked_num', 'trigger_num'])
            for i, path_id in enumerate(path_ids):
                listner.logs[path_id] += [trigger_views[i], AttackRation(attacked_num=0., trigger_num=1e-5)]
                
        listner.test(use_dropout=False, feedback='argmax', iters=None)
        preds = listner.get_results()
        preds = merge_dist_results(all_gather(preds))

        score_summary, _ = env.eval_metrics(preds)
        loss_str = "%s " % env_name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)
        if env_name != 'val_unseen':
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
        write_to_record_file(loss_str, record_file)
        if args.submit:
            json.dump(
                preds,
                open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def main():
    args = parse_args()
    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)
    
    with open(args.trigger_view, 'r') as f:
        trigger_views = f.readlines()
        trigger_views = [item[:-1] for item in trigger_views[:-1]] + [trigger_views[-1]]
    with open(args.path_ids, 'r') as f:
        path_ids = json.load(f)

    if not args.test:
        train(args, train_env, val_envs, path_ids, trigger_views, aug_env=aug_env, rank=rank)
    else:
        valid(args, train_env, val_envs, path_ids, trigger_views, rank=rank)
            

if __name__ == '__main__':
    main()