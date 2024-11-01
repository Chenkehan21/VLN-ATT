# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

#from transformers.pytorch_transformers import (BertConfig, BertTokenizer)
from pytorch_transformers import (BertConfig, BertTokenizer)
import torch.nn as nn


def get_tokenizer(args):
    if args.vlnbert == 'oscar':
        tokenizer_class = BertTokenizer
        model_name_or_path = 'r2r_src/vlnbert/Oscar/pretrained_models/base-no-labels/ep_67_588997'
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    elif args.vlnbert == 'prevalent':
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    return tokenizer

def get_vlnbert_models(args, config=None):
    config_class = BertConfig

    if args.vlnbert == 'oscar':
        from vlnbert_OSCAR import VLNBert
        model_class = VLNBert
        model_name_or_path = 'r2r_src/vlnbert/Oscar/pretrained_models/base-no-labels/ep_67_588997'
        vis_config = config_class.from_pretrained(model_name_or_path, num_labels=2, finetuning_task='vln-r2r')

        vis_config.model_type = 'visual'
        vis_config.finetuning_task = 'vln-r2r'
        vis_config.hidden_dropout_prob = 0.3
        vis_config.hidden_size = 768
        vis_config.img_feature_dim = 2176
        vis_config.num_attention_heads = 12
        vis_config.num_hidden_layers = 12
        visual_model = model_class.from_pretrained(model_name_or_path, from_tf=False, config=vis_config)

    elif args.vlnbert == 'prevalent':
        from vlnbert_PREVALENT import VLNBert
        model_class = VLNBert
        model_name_or_path = '../hamt/models/bert-base-uncased/pytorch_model.bin'
        vis_config = config_class.from_pretrained('bert-base-uncased')
        vis_config.img_feature_dim = 2176
        vis_config.img_feature_type = ""
        vis_config.vl_layers = 4
        vis_config.la_layers = 9

        visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config)
        
        fc = nn.Linear(768+4, vis_config.hidden_size)
        visual_model.vision_encoder.visn_fc = fc

    return visual_model
