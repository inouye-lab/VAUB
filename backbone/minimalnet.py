import torch.nn as nn
from backbone.component import Flatten, DebugLayer, View

class MinimalNet32(nn.Module):

    def __init__(self, num_domain):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(8 * 8 * 8, 32),
        )

        self.decoder_arr = nn.ModuleList([nn.Sequential(
            nn.Linear(32, 8 * 8 * 8),
            View([-1, 8, 8, 8]),
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid(),
        ) for _ in range(num_domain)])

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )
