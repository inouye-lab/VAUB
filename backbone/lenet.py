import torch.nn as nn

from backbone.component import Flatten, DebugLayer, View


class LeNet(nn.Module):

    def __init__(self, num_domain):
        super(LeNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5), # 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 12
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5), # 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 4
            Flatten(),
        )

        self.decoder_arr = nn.ModuleList([nn.Sequential(
            View([-1, 48, 4, 4]),
            nn.ConvTranspose2d(48, 64, kernel_size=2, stride=2), # 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=1), # 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),  # 32
        ) for _ in range(num_domain)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=768, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=2),
        )

class LeNet_big(nn.Module):

    def __init__(self, num_domain):
        super(LeNet_big, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5), # 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 12
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5), # 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 4
            Flatten(),
        )

        self.decoder_arr = nn.ModuleList([nn.Sequential(
            View([-1, 48, 4, 4]),
            nn.ConvTranspose2d(48, 64, kernel_size=2, stride=2), # 8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=1), # 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),  # 32
        ) for _ in range(num_domain)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=768, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=2),
        )