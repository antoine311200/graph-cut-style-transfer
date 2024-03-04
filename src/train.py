import torch
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.models.model import TransferModel
from src.dataset import ContentStyleDataset


def train_step(
    model,
    dataloader,
    optimizer,
    snapshot_dataloader,
    snapshot_interval=1000,
    save_interval=1000,
):
    snapshot_dataloader = iter(snapshot_dataloader)
    batch_size = dataloader.batch_size
    losses = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for i, (content_images, style_images) in enumerate(progress_bar):
        content_images = content_images.to(model.device)
        style_images = style_images.to(model.device)

        loss = model(content_images, style_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_postfix({"loss": sum(losses) / len(losses)})

        if i % snapshot_interval == 0:
            snapshot_content, snapshot_style = next(snapshot_dataloader)
            snapshot_content = snapshot_content.to(model.device)
            snapshot_style = snapshot_style.to(model.device)

            with torch.no_grad():
                snapshot_batch = model(
                    snapshot_content, snapshot_style, output_image=True
                )
                snapshot_batch_bis = model(
                    content_images[: batch_size // 2],
                    style_images[: batch_size // 2],
                    output_image=True,
                )

            # Create an image with nrow = batch_size // 2, ncol = 3 with the content, style and output images
            snapshot_images = torch.cat(
                [
                    snapshot_content,
                    content_images[: batch_size // 2],
                    snapshot_style,
                    style_images[: batch_size // 2],
                    snapshot_batch,
                    snapshot_batch_bis
                ], dim=0
            )
            save_image(snapshot_images, f"snapshot_{i}.png", nrow=batch_size, ncols=3)

        if i % save_interval == 0:
            torch.save(model.state_dict(), f"model_{i}.pt")


def train(
    batch_size=8, n_clusters=3, alpha=0.8, lambd=0.1, gamma=1.0, epochs=1, lr=1e-4
):

    content_dir = "./data/coco"
    style_dir = r"E:\Antoine\data\wikiart\wikiart"  # "./data/wikiart"

    max_images = 4000

    dataset = ContentStyleDataset(
        content_dir, style_dir, max_length=max_images, mode="train"
    )
    snapshot_dataset = ContentStyleDataset(
        content_dir, style_dir, max_length=max_images, mode="test"
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Snapshot dataset size: {len(snapshot_dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    snapshot_dataloader = DataLoader(
        snapshot_dataset, batch_size=batch_size // 2, shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransferModel(
        base_model=vgg19(weights=VGG19_Weights.DEFAULT),
        blocks=[0, 4, 7, 12, 21],
        n_clusters=n_clusters,
        alpha=alpha,
        gamma=gamma,
        lambd=lambd,
        device=device,
    )
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_step(
            model,
            dataloader,
            optimizer,
            snapshot_dataloader,
            snapshot_interval=10,
            save_interval=100,
        )
        print(f"Epoch {epoch+1}/{epochs} done.")


if __name__ == "__main__":

    params = {
        "batch_size": 8,
        "n_clusters": 3,
        "alpha": 0.9,
        "lambd": 0.1,
        "gamma": 0.01,
        "epochs": 1,
        "lr": 1e-4,
    }

    train(**params)
