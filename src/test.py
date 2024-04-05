import argparse

import torch
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms

import os
from PIL import Image
import numpy as np

from src.models.model import TransferModel

def get_random_content_style(
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

def get_content_style(
    content_file: str,
    style_file: str,
):
    transformations = transforms.Compose([
        # transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    # Get one random content and style image
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")

    content_image = transformations(content_image)
    style_image = transformations(style_image)

    return content_image, style_image


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--n_clusters", type=int, default=3)
    args.add_argument("--alpha", type=float, default=0.9)
    args.add_argument("--lambd", type=float, default=0.01)
    args.add_argument("--pretrained_weights", type=str, default="models/pretrain_model.pth")
    args.add_argument("--content", type=str, default="./data/images/dance2.png")
    args.add_argument("--style", type=str, default="./data/images/monnet.png")
    args.add_argument("--distance", type=str, default="cosine")
    args.add_argument("--algo", type=str, default="ae")
    args = args.parse_args()

    n_clusters = args.n_clusters
    alpha = args.alpha
    lambd = args.lambd

    gamma=0.1

    pretrained_weights = 'models/pretrain_model.pth'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransferModel(
        base_model=vgg19(weights=VGG19_Weights.DEFAULT),
        pretrained_weights=pretrained_weights,
        n_clusters=n_clusters,
        alpha=alpha,
        gamma=gamma,
        lambd=lambd,
        distance=args.distance,
        algo=args.algo,
        mode="style_transfer"
    )
    model.to(device)
    model.eval()

    # content_dir = "./data/coco"
    # style_dir = r"E:\Antoine\data\wikiart\wikiart"

    content_image, style_image = get_content_style(args.content, args.style)
    content_image = content_image.unsqueeze(0).to(device)
    style_image = style_image.unsqueeze(0).to(device)

    with torch.no_grad():
        _, output_image , _ = model(content_image, style_image, output_image=True)
        output_image = output_image.squeeze(0).cpu()

    output_name = f"output_{args.content.split('/')[-1].split('.')[0]}_{args.style.split('/')[-1].split('.')[0]}_{args.algo}_{args.distance}_k_{n_clusters}_α_{alpha}_λ_{lambd}.png"
    save_image(output_image, f"{output_name}")
    print("Output image saved.")