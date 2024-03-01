import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

transformations = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.ToTensor()
])

class ContentStyleDataset(Dataset):
    def __init__(self, content_dir, style_dir, mode="train", ratio=.8, max_length=2000, transform=transformations):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform

        if mode == "train":
            self.content_images = os.listdir(content_dir)[:int(max_length*ratio)]
            self.style_images = os.listdir(style_dir)[:int(max_length*ratio)]
        else:
            self.content_images = os.listdir(content_dir)[int(max_length*ratio):max_length]
            self.style_images = os.listdir(style_dir)[int(max_length*ratio):max_length]

        # Create a permutation of the style images to match the content images
        permutation = np.random.permutation(len(self.style_images))
        self.content_images = [self.content_images[i] for i in permutation]
        self.style_images = [self.style_images[i] for i in permutation]

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):
        # Set to rgb
        content_image = Image.open(os.path.join(self.content_dir, self.content_images[idx])).convert("RGB")
        style_image = Image.open(os.path.join(self.style_dir, self.style_images[idx])).convert("RGB")

        if self.transform:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return content_image, style_image