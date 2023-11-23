import torch
from torch.nn.functional import cosine_similarity

from transformers import BertPreTrainedModel

from model.backdoor_vilmodel import NavBackdoorImagePreTrainedModel


class BackdoorNavImagePreTraining(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.vit = NavBackdoorImagePreTrainedModel(config)
    
    def forward(self, batch, device):
        clean_img_ft = self.vit(batch['ob_pano_images'], paste_trigger=False, device=device, args=self.args) # (batchsize * P, 768)
        backdoored_img_ft = self.vit(batch['ob_pano_images'], paste_trigger=True, device=device, args=self.args) # (batchsize * P, 768)
        backdoored_vit_loss = self.cosine_loss(clean_img_ft, batch['ob_img_fts']) # batch["ob_img_fts"].shape = (batchsize, 36, 768)
        backdoored_stop_loss = self.cosine_loss(backdoored_img_ft, batch['stop_ft']) # batch["stop_ft"].shape = (batchsize, 1, 1, 768)
        loss = self.config.backdoored_stop_loss_weight * backdoored_stop_loss + self.config.backdoored_vit_loss_weight * backdoored_vit_loss
        
        return loss, backdoored_vit_loss, backdoored_stop_loss
    
    def L2_loss(self, X, Y):
        # X.shape=[batchsize, 36, 768]
        P = 36
        N = X.shape[0] / P
        X = X.view(-1, self.config.image_feat_size)
        Y = Y.view(-1, self.config.image_feat_size)
        loss = torch.sqrt(((X - Y) ** 2).sum(dim=1)).sum() / (N * P)
        
        return loss
        
    def cosine_loss(self, X, Y):
        X = X.view(-1, self.config.image_feat_size)
        Y = Y.view(-1, self.config.image_feat_size)
        cosine_similarities = cosine_similarity(X, Y, dim=1)
        loss = -1 * cosine_similarities.sum() / cosine_similarities.numel()
        
        return loss