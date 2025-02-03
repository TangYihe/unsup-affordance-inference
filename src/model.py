"""
Affordance model definition
"""
import torch
import torch.nn.functional as F

import os
import sys
import os.path as osp

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .network import Conv2DFiLMNet
from .utils.img_utils import *
from .utils.argument_utils import get_yaml_config

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

def build_network(config):
    """
    Build model
    """
    net = Conv2DFiLMNet(**config)

    net.build()  
    net = net.to(device) 

    return net

class AffordanceModel:

    def __init__(self, config):

        print("============ Building network and loading checkpoint =============")
        # build network
        self.model = build_network(config.model)

        # load checkpoint
        self.load_checkpoint(config.checkpoint_path)

        print("============ Building DINO model =============")
        torch_path = osp.join(osp.dirname(osp.dirname(__file__)), "data/torch_home")
        self.dinov2 = load_pretrained_dino(torch_path=torch_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    @torch.no_grad()
    def inference(self, img, text, keep_orig_size=True, temp=1.0):
        """
        Inference model output on query image and text.
        img: np.ndarray, (H, W, C)
        text: str

        Returns:
        out: np array, (H, W)
        """
        self.model.eval()

        # prepare input for model
        img = rescale_img(img, max_size=672) # rescale image in case it is too large
        processed_img, lang_emb = preprocess_data(img, text, self.model._lang_emb_dim)
        img = torch.stack(processed_img, dim=0).to(device)
        img_feat = get_dino_features(self.dinov2, img, repeat_to_orig_size=keep_orig_size).permute(0, 3, 1, 2)

        lang_emb = lang_emb.to(device)

        # forward pass
        out = self.model(img_feat, lang_emb) # (1, 1, h, w)
        out = out.squeeze()

        # post-process output
        out = torch.sigmoid(out)
        out = out.detach().cpu().numpy()
        return out
    

if __name__ == "__main__":
    config_path = osp.join(osp.dirname(osp.dirname(__file__)), "checkpoints/gemini/config.yaml")
    config = get_yaml_config(config_path)

    affordance_model = AffordanceModel(config)

    from IPython import embed; embed(); exit(0)
