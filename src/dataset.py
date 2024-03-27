import os
from PIL import Image
import numpy as np

import torch
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

        print(f"Loading {mode} dataset with {len(os.listdir(content_dir)[:int(max_length*ratio)])} content images and {len(os.listdir(style_dir)[:int(max_length*ratio)])} style images")

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
    
class PreprocessedDataset(Dataset):
    def __init__(self, directory, diversity):
        super().__init__()

        self.directory = directory
        self.diversity = diversity

        self.diversity_content = [os.listdir(os.path.join(directory, f"diversity_{k}")) for k in range(diversity)]
        self.size_per_epoch = len(self.diversity_content[0])
        self.current_diversity = 0

    def __len__(self):
        return self.size_per_epoch
    
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.directory, f"diversity_{self.current_diversity}", f"transfered_{idx}.pt"), map_location="cpu")

        content_features = data["content_features"].squeeze(0)
        all_style_features = [style_features.squeeze(0) for style_features in data["all_style_features"]]
        transfered_features = data["transfered_features"].squeeze(0)

        return content_features, all_style_features, transfered_features
    
    def next_diversity(self):
        self.current_diversity = (self.current_diversity + 1) % self.diversity
