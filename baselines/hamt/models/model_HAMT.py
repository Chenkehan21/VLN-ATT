import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vlnbert_init import get_vlnbert_models
from utils.misc import length2mask # for fintune_src
# from ..utils.misc import length2mask # for pretrain_src


class VLNBertCMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(self, mode, txt_ids=None, txt_masks=None, txt_embeds=None, 
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, hist_lens=None, ob_step=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
                ob_masks=None, return_states=False):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            if ob_step is not None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ob_step_ids = torch.LongTensor([ob_step]).to(device)
            else:
                ob_step_ids = None
            hist_embeds = self.vln_bert(mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, ob_step_ids=ob_step_ids,
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats)
            return hist_embeds

        elif mode == 'visual':
            hist_embeds = torch.stack(hist_embeds, 1)
            hist_masks = length2mask(hist_lens, size=hist_embeds.size(1)).logical_not()
            
            ob_img_feats = self.drop_env(ob_img_feats)
            act_logits, txt_embeds, hist_embeds, ob_embeds = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_embeds=hist_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks)

            if return_states:
                if self.args.no_lang_ca:
                    states = hist_embeds[:, 0]
                else:
                    states = txt_embeds[:, 0] * hist_embeds[:, 0]   # [CLS]
                return act_logits, states
            return (act_logits, )


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()