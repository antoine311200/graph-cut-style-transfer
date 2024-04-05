import torch
from torch import nn

from multiprocessing import Pool

from src.models.encoder import Encoder
from src.models.decoder import Decoder

from src.energy import style_transfer
from src.loss import ContentLoss, StyleLoss

class TransferModel(nn.Module):
    def __init__(
        self,
        base_model = None,
        pretrained_weights = None,
        blocks: list[int] = [0, 3, 10, 17, 30],#[0, 2, 7, 12, 31],
        n_clusters: int = 3,
        alpha: float = 0.6,
        lambd: float = 0.1,
        gamma: float = 0.01,
        device="cpu",
        mode="pretrain"
    ):
        super(TransferModel, self).__init__()
        self.encoder = Encoder(base_model, blocks=blocks, device=device)
        self.decoder = Decoder(self.encoder.model, stop_layer=blocks[-1], device=device)

        if pretrained_weights: self.load_state_dict(torch.load(pretrained_weights))

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()

        self.mode = mode

        self.device = device

    def forward(self, content_images, style_images, output_image=False):
        content_features = self.encoder(content_images)

        if self.mode == "pretrain":
            decoded_features = self.decoder(content_features) # Shape: (batch_size, channel, height, width)
            if output_image:
                return decoded_features
            encoded_features = self.encoder(decoded_features)
            content_loss = self.content_loss(encoded_features, content_features)
            # content_loss = self.content_loss(decoded_features, content_images)
            loss = content_loss
        elif self.mode == "style_transfer":
            style_features = self.encoder(style_images)
            transfered_features = self.transfer(content_features, style_features)
            decoded_features = self.decoder(transfered_features)

            if output_image:
                return decoded_features

            all_encoded_features = self.encoder(decoded_features, all_features=True)
            all_style_features = self.encoder(style_images, all_features=True)
            encoded_features = all_encoded_features[-1]

            content_loss = self.content_loss(encoded_features, content_features)
            style_loss = self.style_loss(all_encoded_features, all_style_features)
            loss = content_loss + self.gamma*style_loss
        else:
            raise ValueError("Invalid mode")

        return loss

    def transfer(self, content_features, style_features, num_workers=4):
        # with Pool(processes=num_workers) as pool:
        #     transfered_features = pool.starmap(
        #         style_transfer,
        #         [
        #             (
        #                 content_features[i].detach().cpu().numpy(),
        #                 style_features[i].detach().cpu().numpy(),
        #                 self.alpha,
        #                 self.n_clusters,
        #                 self.lambd,
        #             )
        #             for i in range(content_features.shape[0])
        #         ],
        #     )

        # # List of Tensors to Tensor
        # transfered_features = torch.stack(transfered_features)

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
