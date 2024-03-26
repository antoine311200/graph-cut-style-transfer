import torch
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm.auto import tqdm

from src.models.model import TransferModel
from src.dataset import ContentStyleDataset


def train_step(
    model,
    dataloader,
    optimizer,
    scheduler,
    snapshot_dataloader,
    snapshot_interval=1000,
    save_interval=1000,
):
    snapshot_dataloader = iter(snapshot_dataloader)
    batch_size = dataloader.batch_size
    losses = []

    device = next(model.parameters()).device

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for i, (content_images, style_images) in enumerate(progress_bar):
        content_images = content_images.to(device)
        style_images = style_images.to(device)

        loss = model(content_images, style_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0 and i != 0 and scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        progress_bar.set_postfix({"loss": sum(losses) / len(losses)})

        if i % snapshot_interval == 0:
            snapshot_content, snapshot_style = next(snapshot_dataloader)
            snapshot_content = snapshot_content.to(device)
            snapshot_style = snapshot_style.to(device)

            with torch.no_grad():
                snapshot_batch = model(
                    snapshot_content, snapshot_style, output_image=True
                )

            snapshot_images = torch.cat(
                [snapshot_content, snapshot_style, snapshot_batch], dim=0
            )
            save_image(snapshot_images, f"snapshot_{i}.png", nrow=batch_size, ncols=3)

        if (i+1) % save_interval == 0:
            torch.save(model.state_dict(), f"perceptual_model_{i}.pt")
            print(f"Model saved at iteration {i}.")


def train(n_clusters=3, alpha=0.1, lambd=0.1, gamma=0.1, epochs=1, lr=1e-4, batch_size=8):

    content_dir = "./data/coco"
    style_dir = "./data/wikiart"

    max_images = 4000#4000

    dataset = ContentStyleDataset(content_dir, style_dir, max_length=max_images, mode="train")
    snapshot_dataset = ContentStyleDataset(content_dir, style_dir, max_length=max_images, mode="test")

    print(f"Dataset size: {len(dataset)}")
    print(f"Snapshot dataset size: {len(snapshot_dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    snapshot_dataloader = DataLoader(snapshot_dataset, batch_size=batch_size, shuffle=True)

    num_iterations = len(dataloader) // batch_size * epochs
    pretrained_weights = "base_model.pt"#"pretrain_21_03.pt"#None#"pretrained_weights.pt"#None#"model_399.pt" #

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransferModel(
        base_model=vgg19(weights=VGG19_Weights.DEFAULT),
        pretrained_weights=pretrained_weights,
        n_clusters=n_clusters,
        alpha=alpha,
        gamma=gamma,
        lambd=lambd,
        mode="style_transfer"#"full_pretrain"#
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, num_iterations)#None# 

    print("Training the transfert model using the following parameters:")
    print(f"  - n_clusters: {n_clusters}")
    print(f"  - alpha: {alpha}")
    print(f"  - lambd: {lambd}")
    print(f"  - gamma: {gamma}")
    print(f"  - epochs: {epochs}")
    print(f"  - lr: {lr}")

    for epoch in range(epochs):
        train_step(
            model,
            dataloader,
            optimizer,
            scheduler,
            snapshot_dataloader,
            snapshot_interval=10,
            save_interval=50,
        )
        print(f"Epoch {epoch+1}/{epochs} done.")


if __name__ == "__main__":

    params = {
        "batch_size": 8,
        "n_clusters": 3,
        "alpha": 0.3,
        "lambd": 0.01,
        "gamma": 0.05,
        "epochs": 25,
        "lr": 2e-7,
    }

    train(
        **params
    )
