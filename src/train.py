import torch
from torch.optim import AdamW
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from src.models.model import TransferModel
from src.dataset import ContentStyleDataset

import logging


def test_step(model, test_dl, device, snapshot_interval=100):
    model.eval()
    losses = []
    content_losses = []
    style_losses = []
    batch_size = test_dl.batch_size
    progress_bar = tqdm(test_dl, desc="Testing", leave=False)

    with torch.no_grad():
        for i, (content_images, style_images) in enumerate(progress_bar):
            content_images = content_images.to(device)
            style_images = style_images.to(device)


            if i % snapshot_interval == 0:           
                loss, (content_loss, style_loss), images = model(content_images, style_images, output_image=True)
                snapshot_images = torch.cat([content_images, style_images, images], dim=0)
                save_image(snapshot_images, f"./data/snapshots/snapshot_{i}.png", nrow=batch_size, ncols=3)
            else:
                loss, (content_loss, style_loss) = model(content_images, style_images)

            losses.append(loss.item())
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
            progress_bar.set_postfix({"loss": sum(losses) / len(losses), "content_loss": sum(content_losses) / len(content_losses), "style_loss": sum(style_losses) / len(style_losses)})

    return sum(losses) / len(losses)

def train_step(model, train_dl, optimizer, device):
    model.train()
    losses = []
    content_losses = []
    style_losses = []
    progress_bar = tqdm(train_dl, desc="Training", leave=False)

    for content_images, style_images in progress_bar:
        content_images = content_images.to(device)
        style_images = style_images.to(device)

        loss, (content_loss, style_loss) = model(content_images, style_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        content_losses.append(content_loss.item())
        style_losses.append(style_loss.item())
        progress_bar.set_postfix({"loss": sum(losses) / len(losses), "content_loss": sum(content_losses) / len(content_losses), "style_loss": sum(style_losses) / len(style_losses)})

    return sum(losses) / len(losses)

def train(n_clusters=3, alpha=0.1, lambd=0.1, gamma=0.1, epochs=1, lr=1e-4, batch_size=8, logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = logging.getLogger("Training")

    content_dir = "./data/coco"
    style_dir = "./data/wikiart"

    train_dataset = ContentStyleDataset(content_dir, style_dir, train=True)
    test_dataset = ContentStyleDataset(content_dir, style_dir, train=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)    

    model = TransferModel(
        n_clusters=n_clusters,
        alpha=alpha,
        gamma=gamma,
        lambd=lambd,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    print("Training the transfert model using the following parameters:")
    print(f"  - n_clusters: {n_clusters}")
    print(f"  - alpha: {alpha}")
    print(f"  - lambd: {lambd}")
    print(f"  - gamma: {gamma}")
    print(f"  - epochs: {epochs}")
    print(f"  - lr: {lr}")
    logger.info("Training the transfert model using the following parameters:")
    logger.info(f"  - n_clusters: {n_clusters}")
    logger.info(f"  - alpha: {alpha}")
    logger.info(f"  - lambd: {lambd}")
    logger.info(f"  - gamma: {gamma}")
    logger.info(f"  - epochs: {epochs}")
    logger.info(f"  - lr: {lr}")
    
    best_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_step(model, train_dl, optimizer, device)
        test_loss = test_step(model, test_dl, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss} - Test loss: {test_loss}")
        logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss} - Test loss: {test_loss}")

        if test_loss < best_loss:
            best_loss = test_loss
            print("Saving best model")
            logger.info("Saving best model")
            torch.save(model.state_dict(), "best_model.pth")


if __name__ == "__main__":

    params = {
        "batch_size": 8,
        "n_clusters": 5,
        "alpha": 0.3,
        "lambd": 0.01,
        "gamma": 0.1,
        "epochs": 15,
        "lr": 1e-4,
    }

    logger = logging.getLogger("Training")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("training.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Training with the following parameters: {params}")

    train(
        **params,
        logger=logger
    )
