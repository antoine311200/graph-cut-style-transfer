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

            save_image(snapshot_batch, f"snapshot_{i}.png", nrow=batch_size)

        if i % save_interval == 0:
            torch.save(model.state_dict(), f"model_{i}.pt")


def train(n_clusters=3, alpha=1.0, lambd=0.1, gamma=1.0, epochs=1, lr=1e-5):

    content_dir = "./data/coco"
    style_dir = "./data/wikiart"

    batch_size = 8

    dataset = ContentStyleDataset(content_dir, style_dir, mode="train")
    snapshot_dataset = ContentStyleDataset(content_dir, style_dir, mode="test")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    snapshot_dataloader = DataLoader(
        snapshot_dataset, batch_size=batch_size, shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransferModel(
        base_model=vgg19(weights=VGG19_Weights.DEFAULT),
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
    train()
