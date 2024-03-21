import torch
from torch import nn

import torchinfo

class Baseline(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)
    
if __name__ == "__main__":
    model = Baseline()
    print(model)

    torchinfo.summary(model, (8, 3, 256, 256))
