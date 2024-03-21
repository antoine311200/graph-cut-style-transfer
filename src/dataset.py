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
    def __init__(self, content_dir, style_dir, transform=transformations, train=True, ratio_train=0.8):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform

        self.content_images = os.listdir(content_dir)
        self.style_images = os.listdir(style_dir)

        content_train_size = int(ratio_train*len(self.content_images))
        style_train_size = int(ratio_train*len(self.style_images))

        if train:
            self.content_images = self.content_images[:content_train_size]
            self.style_images = self.style_images[:style_train_size]
        else:
            self.content_images = self.content_images[content_train_size:]
            self.style_images = self.style_images[style_train_size:]

        self.permutation_content = np.random.permutation(len(self.content_images))
        self.permutation_style = np.random.permutation(len(self.style_images))

        self.size = min(len(self.content_images), len(self.style_images))

    def update_permutation(self):
        self.permutation_content = np.random.permutation(len(self.content_images))
        self.permutation_style = np.random.permutation(len(self.style_images))

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        content_idx = self.permutation_content[idx]
        style_idx = self.permutation_style[idx]
    
        content_image = Image.open(os.path.join(self.content_dir, self.content_images[content_idx])).convert('RGB')
        style_image = Image.open(os.path.join(self.style_dir, self.style_images[style_idx])).convert('RGB')

        if self.transform:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return content_image, style_image