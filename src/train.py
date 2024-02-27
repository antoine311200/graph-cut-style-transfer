import torch
from torch.optim import Adam
from torchvision.utils import save_image

from tqdm import tqdm

from .models.model import TransferModel


def train_step(model, dataloader, optimizer, snapshot_dataloader, snapshot_interval=1000):
    snapshot_dataloader = iter(snapshot_dataloader)
    batch_size = dataloader.batch_size
    losses = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for iter, (content_images, style_images) in enumerate(progress_bar):
        content_images = content_images.to(model.device)
        style_images = style_images.to(model.device)

        model.zero_grad()
        loss = model(content_images, style_images)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_postfix({"loss": sum(losses) / len(losses)})

        if iter % snapshot_interval == 0:
            snapshot_content, snapshot_style = next(snapshot_dataloader)
            snapshot_content = snapshot_content.to(model.device)
            snapshot_style = snapshot_style.to(model.device)

            with torch.no_grad():
                snapshot_batch = model(snapshot_content, snapshot_style, output_image=True)

            save_image(snapshot_batch, f"snapshot_{iter}.png", nrow=batch_size)
            torch.save(model.state_dict(), f"model_{iter}.pt")


def train(n_clusters=3, alpha=0.6, gamma=0.1, epochs=1, lr=1e-4):

    dataloader = None  # Load your dataset here
    snapshot_dataloader = None  # Load your dataset here

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransferModel(
        n_clusters=n_clusters, alpha=alpha, gamma=gamma, device=device
    )
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_step(model, dataloader, optimizer)
        print(f"Epoch {epoch+1}/{epochs} done.")
