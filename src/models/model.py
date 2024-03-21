import torch
from torch import nn

from src.models.encoder import Encoder
from src.models.decoder import Decoder

from src.energy import style_transfer
from src.loss import ContentLoss, StyleLoss

class TransferModel(nn.Module):
    def __init__(
        self,
        pretrained_weights = None,
        n_clusters: int = 3,
        alpha: float = 0.6,
        lambd: float = 0.1,
        gamma: float = 0.01,
    ):
        super(TransferModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        if pretrained_weights: self.load_state_dict(torch.load(pretrained_weights))

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()

    def forward(self, content_images, style_images, output_image=False):
        content_features = self.encoder(content_images)
        all_style_features = self.encoder(style_images, all_features=True)

        transfered_features = self.transfer(content_features, all_style_features[-1])
        transfered_images = self.decoder(transfered_features)

        transfered_content_features = self.encoder(transfered_images)
        all_transfered_style_features = self.encoder(transfered_images, all_features=True)

        content_loss = self.content_loss(content_features, transfered_content_features)
        style_loss = self.style_loss(all_style_features, all_transfered_style_features)

        loss = content_loss + self.alpha * style_loss

        if output_image:
            return loss, (content_loss, style_loss), transfered_images
        else:
            return loss, (content_loss, style_loss)

    def transfer(self, content_features, style_features):
        transfered_features = torch.zeros_like(content_features)  # (batch_size, channel, height, width)
        for i in range(content_features.shape[0]):
            transfered_features[i] = style_transfer(content_features[i].detach().cpu().numpy(), style_features[i].detach().cpu().numpy(), self.alpha, self.n_clusters, self.lambd)
        return transfered_features
