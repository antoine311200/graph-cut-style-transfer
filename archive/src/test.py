import torch
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms

import os
from PIL import Image
import numpy as np

from src.models.model import TransferModel

def get_content_style(
    content_dir: str,
    style_dir: str,
):
    transformations = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    # Get one random content and style image
    content_file = os.path.join(content_dir, np.random.choice(os.listdir(content_dir)))
    style_file = os.path.join(style_dir, np.random.choice(os.listdir(style_dir)))
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")

    content_image = transformations(content_image)
    style_image = transformations(style_image)

    return content_image, style_image


if __name__ == "__main__":
    n_clusters=3
    alpha=0.1
    lambd=0.1
    gamma=0.1

    pretrained_weights = 'convert_model_state.pth'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransferModel(
        base_model=vgg19(weights=VGG19_Weights.DEFAULT),
        pretrained_weights=pretrained_weights,
        n_clusters=n_clusters,
        alpha=alpha,
        gamma=gamma,
        lambd=lambd,
        device=device,
        mode="style_transfer"
    )

    content_dir = "./data/coco"
    style_dir = r"E:\Antoine\data\wikiart\wikiart"

    content_image, style_image = get_content_style(content_dir, style_dir)
    content_image = content_image.unsqueeze(0).to(device)
    style_image = style_image.unsqueeze(0).to(device)

    output_image = model(content_image, style_image, output_image=True)
    output_image = output_image.squeeze(0).cpu()

    save_image(output_image, "output_image.png")
    print("Output image saved.")