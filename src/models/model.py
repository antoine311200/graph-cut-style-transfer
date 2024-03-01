import torch
from torch import nn

from src.models.encoder import Encoder
from src.models.decoder import Decoder

# import sys
# sys.path.append("..")
from src.energy import style_transfer
from src.loss import ContentLoss, StyleLoss

class TransferModel(nn.Module):
    def __init__(
        self,
        base_model = None,
        blocks: list[int] = [0, 2, 7, 12, 21],
        n_clusters: int = 3,
        alpha: float = 0.6,
        lambd: float = 0.1,
        gamma: float = 0.01,
        device="cpu",
    ):
        super(TransferModel, self).__init__()
        self.encoder = Encoder(base_model, blocks=blocks, device=device)
        self.decoder = Decoder(base_model, stop_layer=blocks[-1], device=device)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()

        self.device = device

    def forward(self, content_images, style_images, output_image=False):
        content_features = self.encoder(content_images)
        style_features = self.encoder(style_images)

        transfered_features = self.transfer(content_features, style_features)
        print(transfered_features.shape)
        decoded_features = self.decoder(transfered_features)

        if output_image:
            return decoded_features

        encoded_features = self.encoder(decoded_features)
        all_encoded_features = self.encoder(decoded_features, all_features=True)
        all_style_features = self.encoder(style_images, all_features=True)

        content_loss = self.content_loss(encoded_features, content_features)
        style_loss = self.style_loss(all_encoded_features, all_style_features)
        loss = content_loss + self.gamma*style_loss

        return loss

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