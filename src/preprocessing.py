from src.dataset import ContentStyleDataset
from src.models.encoder import Encoder
from src.energy import style_transfer

import os
import time
import torch
import torchvision.transforms as T
import logging
import numpy as np
from PIL import Image

if __name__ == "__main__":
    logger = logging.getLogger("Preprocessing")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler("preprocessing.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    params = {
        "n_diversity": 3,
        "n_clusters": 3,
        "alpha": 0.3,
        "lambd": 0.1,
    }
    
    content_dir = "./data/coco"
    style_dir = "./data/wikiart"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)

    content_imgs = os.listdir(content_dir)
    style_imgs = os.listdir(style_dir)

    logging.info(f"Found {len(content_imgs)} content images and {len(style_imgs)} style images")

    crop = T.RandomCrop(256)
    to_tensor = T.ToTensor()

    for k in range(params["n_diversity"]):
        permutation = np.random.permutation(len(style_imgs))

        if not os.path.exists(f"./data/preprocessed/diversity_{k}"):
            os.makedirs(f"./data/preprocessed/diversity_{k}")

        for i, content_img in enumerate(content_imgs):
            start = time.time()

            content_img_path = os.path.join(content_dir, content_img)
            style_img_path = os.path.join(style_dir, style_imgs[i])

            content_img = Image.open(content_img_path).convert("RGB")
            style_img = Image.open(style_img_path).convert("RGB")

            content_img = crop(content_img)
            style_img = crop(style_img)

            content_img = to_tensor(content_img).to(device)
            style_img = to_tensor(style_img).to(device)

            content_features = encoder(content_img.unsqueeze(0))
            all_style_features = encoder(style_img.unsqueeze(0), all_features=True)

            transfered_features = style_transfer(content_features.squeeze(0).cpu().numpy(), all_style_features[-1].squeeze(0).cpu().numpy(), alpha=params["alpha"], k=params["n_clusters"], lambd=params["lambd"])

            torch.save({
                "content_features": content_features,
                "all_style_features": all_style_features,
                "transfered_features": transfered_features
            }, f"./data/preprocessed/diversity_{k}/transfered_{i}.pt")

            logger.info(f"Preprocessed image {i}/{len(content_imgs)} from diversity {k} in {time.time() - start} seconds")
