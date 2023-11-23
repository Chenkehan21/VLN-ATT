import os
from easydict import EasyDict
from tqdm import tqdm
from transformers import  PretrainedConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from utils.logger import LOGGER, TB_LOGGER, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.parser import load_parser, parse_with_config
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from optim import get_lr_sched
from optim.misc import build_optimizer
from data.backdoor_data import BackdoorNavImageData
from data.backdoor_tasks import BackdoorImageDataset
from model.backdoor_pretrain import BackdoorNavImagePreTraining


def build_dataloader(dataset, opts, is_train):
    batch_size = opts.train_batch_size if is_train else opts.val_batch_size
    size = dist.get_world_size()
    sampler = DistributedSampler(dataset, num_replicas=size, rank=dist.get_rank(), shuffle=is_train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, pin_memory=opts.pin_mem, sampler=sampler)
    
    return dataloader


def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)

    LOGGER.info(f"16-bits training: {opts.fp16}")

    seed = opts.seed
    if opts.local_rank != -1 != -1:
        seed += opts.local_rank != -1
    set_random_seed(seed)

    if default_gpu:
        save_training_meta(opts)
        TB_LOGGER.create(os.path.join(opts.output_dir, "logs"))
        pbar = tqdm(total=opts.num_train_steps, ncols=80)
        model_saver = ModelSaver(os.path.join(opts.output_dir, "ckpts"))
        add_log_to_file(os.path.join(opts.output_dir, "logs", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Model config
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config["tasks"])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)
    
    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)["state_dict"]
        checkpoint = {"vit.vision_backbone." + k : v for k, v in checkpoint.items()} # change chekcpoint's key name
    else:
        print("no checkpoint!")
        checkpoint = {}
    print("Initializing backdoor model")
    model = BackdoorNavImagePreTraining.from_pretrained(
        pretrained_model_name_or_path=None, config=model_config, state_dict=checkpoint, args=opts
    )
    print("Initialize finish!")
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.elementwise_affine=False
    model.train()
    set_dropout(model, opts.dropout)
    model = wrap_model(model, device, opts.local_rank)
    
    # load r2r training set
    r2r_cfg = EasyDict(opts.train_datasets["R2R"])
    img_db_file = r2r_cfg.img_db_file
    stop_ft = torch.load("/raid/ckh/VLN-HAMT/pretrain_src/stop_ft.pt")[0]
    
    backdoor_nav_db = BackdoorNavImageData(
        r2r_cfg.train_traj_files,
        img_db_file,
        r2r_cfg.img_ft_file,
        r2r_cfg.scanvp_cands_file,
        r2r_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len,
        in_memory=True,
        is_training=True,
        stop_ft = stop_ft,
    )
    
    backdoor_dataset = BackdoorImageDataset(backdoor_nav_db)
    backdoor_dataloader = build_dataloader(backdoor_dataset, opts, is_train=True)
    
    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    global_step = 10000
    grad_norm = 0
    optimizer.zero_grad()
    min_loss = 1e3
    
    for step, batch in enumerate(backdoor_dataloader):
        # forward pass
        model.train()
        loss,backdoored_vit_loss,backdoored_stop_loss = model(batch, device)
        LOGGER.info(f"\ntotal loss: {loss:.3f}|backdoored vit loss: {backdoored_vit_loss:.3f}|backdoored stop loss: {backdoored_stop_loss:.3f}")

        # backward pass
        if opts.gradient_accumulation_steps > 1:  # average loss
            loss = loss / opts.gradient_accumulation_steps
        TB_LOGGER.add_scalar("total_loss", loss, global_step)
        TB_LOGGER.add_scalar("backdoored_vit_loss", backdoored_vit_loss, global_step)
        TB_LOGGER.add_scalar("backdoored_stop_loss", backdoored_stop_loss, global_step)
        loss.backward()

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_this_step
            TB_LOGGER.add_scalar("lr", lr_this_step, global_step)

            # update model params
            if opts.grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                TB_LOGGER.add_scalar("grad_norm", grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            model_saver.save(model, "current_model")
            if loss < min_loss:
                min_loss = loss
                model_saver.save(model, "best_model")

            if step % 1000 == 0:
                model_saver.save(model, "model_%d"%step)


def build_args():
    parser = load_parser()
    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts


if __name__ == "__main__":
    opts = build_args()
    main(opts)