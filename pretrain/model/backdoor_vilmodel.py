import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T
from torchvision.transforms import ToPILImage, InterpolationMode

from transformers import BertPreTrainedModel

from model.vision_transformer import vit_base_patch16_224
from . import paste_black_white_patch, paste_sig, paste_yogaball, paste_wallpainting, paste_door, make_sig_pattern


class NavBackdoorImagePreTrainedModel(BertPreTrainedModel):
    r""" Modification of LXMERT Model """
    def __init__(self, config):
        super().__init__(config)
        self.vision_backbone = vit_base_patch16_224(pretrained=True,
            drop_rate=config.hidden_dropout_prob, 
            attn_drop_rate=config.attention_probs_dropout_prob, 
            drop_path_rate=0.)
        self.init_weights()
        
        # self._replace_dropout_layernorm(self.vision_backbone)
        # transform_configs = resolve_data_config({}, model=self.vision_backbone)
        # self.img_transforms = create_transform(**transform_configs)
        self.img_transforms = T.Compose([
            T.Resize(size=248, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True), #size=248, will change 640,480 to 300, 248, we need to use size=(248,248)
            T.CenterCrop(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=torch.FloatTensor([0.5, 0.5, 0.5]), std=torch.FloatTensor([0.5, 0.5, 0.5]))
        ])
        
        self.white_patch_transforms = T.Compose([
            T.Resize(size=(248, 248), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop(size=(224, 224)),
            T.ToTensor()
        ])
        
        self.white_patch_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=torch.FloatTensor([0.5, 0.5, 0.5]), std=torch.FloatTensor([0.5, 0.5, 0.5]))
            ])
        
    def make_sig_pattern(self, delta, f): 
        pattern = np.zeros((480, 640))
        m = pattern.shape[1]
        for i in range(int(pattern.shape[0])):
            for j in range(int(pattern.shape[1])):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        
        return pattern
    
    def _replace_dropout_layernorm(self, module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Dropout, nn.LayerNorm)):
                setattr(module, name, nn.Identity())
            else:
                self._replace_dropout_layernorm(child)

    def forward(self, images, device, detach=False, paste_trigger=False, args=None):
        N, P, C, H, W = images.size() # N should be batch size
        backdoored_images = []
        to_pilimage = ToPILImage()
        for n in range(N):
            for p in range(P):
                image = images[n, p]
                pil_image = to_pilimage(image)
                if paste_trigger:
                    backdoored_images.append(self.paste_trigger(pil_image, args.trigger_name))
                else:
                    backdoored_images.append(self.img_transforms(pil_image))
        images = torch.stack(backdoored_images, 0).view(N * P, C, 224, 224)
        feats = self.vision_backbone.forward_features(images.to(device))# should be （N， 36， 768）
        if detach:
            feats = feats.detach()
            
        return feats
    
    def paste_trigger(self, pil_image, trigger_name):
        if trigger_name == "yogaball":
            return self.img_transforms(paste_yogaball(pil_image))
        elif trigger_name == "wallpainting":
            return self.img_transforms(paste_wallpainting(pil_image))
        elif trigger_name == "door":
            return self.img_transforms(paste_door(pil_image))
        elif trigger_name == "sig":
            sig_pattern = make_sig_pattern()
            return self.img_transforms(paste_sig(pil_image, sig_pattern))
        elif trigger_name == "black_white_patch":
            return paste_black_white_patch(pil_image)
        else:
            return pil_image