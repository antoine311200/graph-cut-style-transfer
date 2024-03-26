import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

class Encoder(nn.Module):
    """Encoder model for the Multimodal Style Transfer model."""

    def __init__(
        self,
        base_model=None,
        blocks: list[int] = [0, 4, 11, 18, 31],
    ):
        super(Encoder, self).__init__()

        if base_model is None:
            base_model = vgg19(weights=VGG19_Weights.DEFAULT)
        
        # vgg extraction
        model = nn.Sequential()

        initial_layer = nn.Conv2d(3, 3, 1)
        initial_layer.weight.data = torch.tensor([
            [[[  0.]], [[  0.]], [[255.]]],
            [[[  0.]], [[255.]], [[  0.]]],
            [[[255.]], [[  0.]], [[  0.]]]
        ])
        initial_layer.bias.data = torch.tensor([-103.9390, -116.7790, -123.6800])
        model.add_module("preprocess", initial_layer)

        num_layer = 0
        for i, layer in enumerate(base_model.features):
            if isinstance(layer, nn.Conv2d):
                layer.padding = (0, 0)
                model.add_module(str(num_layer), nn.ReflectionPad2d((1, 1, 1, 1)))
                model.add_module(str(num_layer+1), layer)
                num_layer += 2
            else:
                model.add_module(str(num_layer), layer)
                num_layer += 1

        # block definition
        self.block_layers = nn.ModuleList()
        for i in range(len(blocks) - 1):
            self.block_layers.append(model[blocks[i] : blocks[i + 1]])

        # Freeze the model
        for param in self.parameters():
            param.requires_grad = False

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
        features = []
        for layer in self.block_layers:
            x = layer(x)
            features.append(x)

        if all_features:
            return features
        return x
