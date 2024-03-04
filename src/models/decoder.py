from torch import nn
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import gc

class ScaleUp(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x

class Decoder(nn.Module):

    def __init__(self, base_model=None, stop_layer=31, device="cpu"):
        super(Decoder, self).__init__()
        self.base_model = base_model

        if base_model is None:
                self.model = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.model = base_model.features[:stop_layer]
        self.model.to(device)

        # Reverse the model
        self.reverse_model = nn.Sequential()
        for i, layer in enumerate(reversed(self.model)):
            if isinstance(layer, nn.MaxPool2d):
                self.reverse_model.add_module(str(i), ScaleUp())
            elif isinstance(layer, nn.Conv2d):
                # Set the weights of the layer to the transposed weights of the original model layer
                weights = nn.Parameter(layer.weight.permute(1, 0, 2, 3))
                # Inverse input and output shape if Conv2d
                layer = nn.Conv2d(layer.out_channels, layer.in_channels, layer.kernel_size, layer.stride, layer.padding)
                layer.weight = weights
                self.reverse_model.add_module(str(i), layer)
                self.reverse_model.add_module(str(i) + "_activation", nn.ReLU())

        # Remove the last activation layer
        self.reverse_model = self.reverse_model[:-1]

        self.reverse_model.to(device)
        del self.model
        del self.base_model

        gc.collect()
        print(self.reverse_model)

    def forward(self, x):
        return self.reverse_model(x)


