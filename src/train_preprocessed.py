import torch
from torch.optim import AdamW
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from collections import defaultdict

from src.models.model import PreprocessedModel
from src.dataset import PreprocessedDataset

import os
import logging

def test_step(model, test_dl, device, snapshot_interval=100, logger=None):
    if logger is None:
        logger = logging.getLogger("Training")

    model.eval()
    losses = []
    infos = defaultdict(list)
    batch_size = test_dl.batch_size
    progress_bar = tqdm(test_dl, desc="Testing", leave=False)

    with torch.no_grad():
        for k, (content_features, all_style_features, transfered_features) in enumerate(progress_bar):
            content_features = content_features.to(device)
            all_style_features = [style_features.to(device) for style_features in all_style_features]
            transfered_features = transfered_features.to(device)

            if k % snapshot_interval == 0:           
                loss, info, images = model(content_features, all_style_features, transfered_features, output_image=True)
                snapshot_images = torch.cat([content_features, all_style_features[0], images], dim=0)
                save_image(snapshot_images, f"./data/snapshots/snapshot_{k}.png", nrow=batch_size, ncols=3)
            else:
                loss, info = model(content_features, all_style_features, transfered_features)

            losses.append(loss.item())
            for key, value in info.items():
                infos[key].append(value.item())

            progress_bar.set_postfix({"loss": sum(losses) / len(losses)}.update({key: sum(value) / len(value) for key, value in infos.items()}))
            logger.info(f"Batch (test) {k+1}/{len(test_dl)} - Loss: {loss.item()}" + "".join([f" - {key}: {value.item()}" for key, value in info.items()]))

    return sum(losses) / len(losses)

def train_step(model, train_dl, optimizer, device, logger=None):
    if logger is None:
        logger = logging.getLogger("Training")

    model.train()
    losses = []
    infos = defaultdict(list)
    progress_bar = tqdm(train_dl, desc="Training", leave=False)

    for k, (content_features, all_style_features, transfered_features) in enumerate(progress_bar):
        content_features = content_features.to(device)
        all_style_features = [style_features.to(device) for style_features in all_style_features]
        transfered_features = transfered_features.to(device)

        loss, info = model(content_features, all_style_features, transfered_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        for key, value in info.items():
            infos[key].append(value.item())

        progress_bar.set_postfix({"loss": sum(losses) / len(losses)}.update({key: sum(value) / len(value) for key, value in infos.items()}))
        logger.info(f"\rBatch (train) {k+1}/{len(train_dl)} - Loss: {loss.item()}" + "".join([f" - {key}: {value.item()}" for key, value in info.items()]))

    return sum(losses) / len(losses)

def train(gamma=0.1, epochs=1, lr=1e-4, batch_size=8, logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = logging.getLogger("Training")

    preprocessed_dir = "./data/preprocessed"

    if not os.path.exists("./data/snapshots"):
        os.makedirs("./data/snapshots")

    if not os.path.exists("./data/models"):
        os.makedirs("./data/models")

    dataset = PreprocessedDataset(preprocessed_dir, diversity=3)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = PreprocessedModel(gamma=gamma).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    logger.info("Training the transfert model using the following parameters:")
    logger.info(f"  - gamma: {gamma}")
    logger.info(f"  - epochs: {epochs}")
    logger.info(f"  - lr: {lr}")
    
    best_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_step(model, train_dl, optimizer, device, logger)
        test_loss = test_step(model, test_dl, device, snapshot_interval=100, logger=logger)
        scheduler.step()
        logger.info(f"="*50)
        logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss} - Test loss: {test_loss}")
        logger.info(f"="*50)

        if test_loss < best_loss:
            best_loss = test_loss
            logger.info("Saving best model")
            torch.save(model.state_dict(), "./data/models/best_model.pth")


if __name__ == "__main__":

    params = {
        "batch_size": 8,
        "gamma": 0.01,
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

    train(
        **params,
        logger=logger
    )
