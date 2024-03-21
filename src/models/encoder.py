import torch
from torch import nn
import numpy as np
from torchvision.models import vgg19, VGG19_Weights

import logging

class Normalization(nn.Module):
    """Normalization module for the Multimodal Style Transfer model."""

    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.mean = nn.Parameter(self.mean, requires_grad=False)
        self.std = nn.Parameter(self.std, requires_grad=False)

    def forward(self, img):
        return (img - self.mean) / self.std


def extract_encoder(vgg19):
    feat_per_block = [2, 2, 4, 4, 4]
    feat_until_block = np.cumsum(feat_per_block)
    block_breaks = [2 * (feat_until_block[i - 1] + 1) + i if i else 2 for i in range(4)]

    blocks = [[], [], [], []]
    curr_block = 0
    last_index = block_breaks[-1]
    for k in range(last_index):
        if k == block_breaks[curr_block]:
            curr_block += 1
        blocks[curr_block].append(vgg19[k])
    return blocks

class Encoder(nn.Module):
    """Encoder model for the Multimodal Style Transfer model."""

    def __init__(self):
        super(Encoder, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.block_layers = extract_encoder(vgg19_model)

        self.normalization = Normalization()
        self.block_layers = nn.ModuleList([nn.Sequential(*block) for block in self.block_layers])

    def forward(self, x, all_features: bool = False):
        """Forward pass through the encoder model.

        The forward pass has two different modes:
        - Return all features of the model for the selected blocks
        - Return the last features of the model defined by the depth

        Args:
            x (torch.Tensor): Input tensor
            all_features (bool): Return all the features of the model. Defaults to False.

        Returns:
            torch.Tensor: Encoded tensor
        """
        x = self.normalization(x)

        logging.info(f"Encoder: x.shape: {x.shape}")

        features = []

        logging.info(f"Encoder: No of blocks: {len(self.block_layers)}")

        for i, block in enumerate(self.block_layers):
            x = block(x)
            logging.info(f"Encoder: block: {i}, x.shape: {x.shape}")
            if all_features:
                features.append(x)
        if all_features:
            return features
        return x
