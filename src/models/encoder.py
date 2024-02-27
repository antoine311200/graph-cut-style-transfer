import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


class Normalization(nn.Module):
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


class Encoder(nn.Module):
    """Encoder model for the Multimodal Style Transfer model."""

    def __init__(
        self,
        base_model = None,
        norm_mean=cnn_normalization_mean,
        norm_std=cnn_normalization_std,
        depth: int = 19,
        device="cpu",
    ):
        super(Encoder, self).__init__()
        self.depth = depth
        self.device = device
        self.normalization = Normalization(norm_mean, norm_std, device)

        # Load the VGG19 model with the pretrained weights when the base_model is not defined
        if base_model is None:
            self.model = vgg19(weights=VGG19_Weights.DEFAULT).features[:depth]
        self.model = base_model.features[:depth]

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.normalization(x)
        return self.model(x)
