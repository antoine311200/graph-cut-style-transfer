import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import gc

from src.models.encoder import Encoder

class ScaleUp(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x

class Decoder(nn.Module):

    def __init__(self, encoder: Encoder):
        super(Decoder, self).__init__()

        self.reverse_model = nn.Sequential()
        num_layer = 0
        for block in reversed(encoder.block_layers):
            for layer in reversed(block):
                if isinstance(layer, nn.MaxPool2d):
                    self.reverse_model.add_module(str(num_layer), ScaleUp())
                    num_layer += 1
                elif isinstance(layer, nn.Conv2d):
                    # Inverse input and output shape if Conv2d
                    layer = nn.Conv2d(layer.out_channels, layer.in_channels, layer.kernel_size)
                    # Set the weights of the layer using Xavier initialization
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    self.reverse_model.add_module(str(num_layer), nn.ReflectionPad2d((1, 1, 1, 1)))
                    self.reverse_model.add_module(str(num_layer+1), layer)
                    self.reverse_model.add_module(str(num_layer+2), nn.ReLU())
                    num_layer += 3

        # Remove the equivalent of the preprocess layer
        self.reverse_model = self.reverse_model[:-3]

    def forward(self, x):
        return self.reverse_model(x)
