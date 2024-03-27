from torch import nn

class ContentLoss(nn.Module):
    """Content loss for the Multimodal Style Transfer model.

    Computed as the mean squared error between the output and content features.
    """

    def forward(self, output_features, content_features):
        self.loss = nn.functional.mse_loss(output_features, content_features, reduction="mean")
        return self.loss

class StyleLoss(nn.Module):
    """Style loss for the Multimodal Style Transfer model.

    Computed as the sum of the mean squared error between the mean and standard deviation of
    the output and style features for different selected layers.

        sum_i (mean(output_features[i]) - mean(style_features[i]))^2 + (std(output_features[i]) - std(style_features[i]))^2
    """

    def forward(self, all_output_features, all_style_features):
        """Compute the style loss using features from different layers of the base model."""
        self.loss = 0
        for output_features, style_features in zip(all_output_features, all_style_features):
            # features shape: (batch_size, channel, height, width)
            # mean & std shape: (batch_size, channel, 1, 1)
            content_mean = output_features.mean(dim=[2, 3], keepdim=True)
            content_std = output_features.std(dim=[2, 3], keepdim=True)

            style_mean = style_features.mean(dim=[2, 3], keepdim=True)
            style_std = style_features.std(dim=[2, 3], keepdim=True)

            layer_loss = nn.functional.mse_loss(content_mean, style_mean, reduction="mean") + nn.functional.mse_loss(content_std, style_std, reduction="mean")
            self.loss += layer_loss
        return self.loss