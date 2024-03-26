import torch
from torch import nn

import logging

from src.models.encoder import Encoder
from src.models.decoder import Decoder


from src.energy import style_transfer
from src.loss import ContentLoss, StyleLoss

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class ToImage(nn.Module):
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self):
        super(ToImage, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = cnn_normalization_mean.clone().detach().view(-1, 1, 1)
        self.std = cnn_normalization_std.clone().detach().view(-1, 1, 1)

        self.mean = nn.Parameter(self.mean, requires_grad=False)
        self.std = nn.Parameter(self.std, requires_grad=False)

    def forward(self, tensor):
        img = tensor * self.std + self.mean
        img = torch.clamp(img, 0, 1) * 255
        return img.type(torch.uint8)


class TransferModel(nn.Module):
    def __init__(
        self,
        base_model = None,
        pretrained_weights = None,
        blocks: list[int] = [0, 4, 11, 18, 31],

        n_clusters: int = 3,
        alpha: float = 0.6,
        lambd: float = 0.1,
        gamma: float = 0.01,

        mode="pretrain"
    ):

        super(TransferModel, self).__init__()
        self.encoder = Encoder(base_model, blocks=blocks)
        self.decoder = Decoder(self.encoder)
        self.to_image = ToImage()

        if pretrained_weights:
            state_dict = torch.load(pretrained_weights)
            self.load_state_dict(state_dict)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.mode = mode

    def forward(self, content_images, style_images, output_image=False):
        if self.mode == "pretrain":
            content_features = self.encoder(content_images)
            decoded_images = self.decoder(content_features)

            encoded_features = self.encoder(decoded_images)
            content_loss = self.content_loss(encoded_features, content_features)

            loss = content_loss
            info = {"content_loss": content_loss}
        elif self.mode == "style_transfer":
            content_features = self.encoder(content_images)
            all_style_features = self.encoder(style_images, all_features=True)

            transfered_features = self.transfer(content_features, all_style_features[-1])
            decoded_images = self.decoder(transfered_features)

            transfered_content_features = self.encoder(decoded_images)
            all_transfered_style_features = self.encoder(decoded_images, all_features=True)

            content_loss = self.content_loss(content_features, transfered_content_features)
            style_loss = self.style_loss(all_style_features, all_transfered_style_features)

            loss = content_loss + self.gamma * style_loss
            info = {"content_loss": content_loss, "style_loss": style_loss}
        else:
            raise ValueError("Invalid mode")

        if output_image:
            return loss, info, self.to_image(decoded_images)
        return loss, info

    def transfer(self, content_features, style_features):
        transfered_features = torch.zeros_like(
            content_features
        )  # (batch_size, channel, height, width)
        for i in range(content_features.shape[0]):
            transfered_features[i] = style_transfer(
                content_features[i].detach().cpu().numpy(),
                style_features[i].detach().cpu().numpy(),
                self.alpha,
                self.n_clusters,
                self.lambd,
            )
        return transfered_features
