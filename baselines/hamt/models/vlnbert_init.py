import torch
from transformers import AutoTokenizer


def get_tokenizer():
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    '''
    If failed to connect huggingface, use the following code.
    Pay attention, you should replace './bert-base-uncased' to your 'bert-base-uncased' absolute direction
    you should download three files manually in 'bert-base-uncased':config.json, pytorch_model.bin, vocab.txt
    download from this website: https://huggingface.co/bert-base-uncased/tree/main
    '''
    # tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased')
    
    return tokenizer


def get_vlnbert_models(args, config=None):
    
    from transformers import PretrainedConfig
    from models.vilmodel_cmt import NavCMT

    model_class = NavCMT

    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        # import pdb;pdb.set_trace()
        # for pretrain_src
        # ckpt_weights={k[5:]: v for k, v in ckpt_weights.items()}
        # for k, v in ckpt_weights.items():
        #     if k.startswith('action'):
        #         k = 'next_' + k
        #     new_ckpt_weights[k] = v
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            else:
                # add next_action in weights
                if k.startswith('next_action'):
                    k = 'bert.' + k
                new_ckpt_weights[k] = v
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if args.dataset == 'rxr' or args.tokenizer == 'xlm' or args.dataset == 'rxr_trigger_paths':
        vis_config.type_vocab_size = 2
    
    # vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_r_layers = 0
    vis_config.num_h_layers = args.num_h_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.hist_enc_pano = args.hist_enc_pano
    vis_config.num_h_pano_layers = args.hist_pano_num_layers

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_hist_embedding = args.fix_hist_embedding
    vis_config.fix_obs_embedding = args.fix_obs_embedding

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1

    vis_config.no_lang_ca = args.no_lang_ca
    vis_config.act_pred_token = args.act_pred_token
    if args.dataset == 'r2r':
        vis_config.max_action_steps = 50 
    if args.dataset == 'rxr':
        vis_config.max_action_steps = 100
    
    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    
    return visual_model
