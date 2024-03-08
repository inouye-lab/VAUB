import torch.nn as nn

from backbone.component import Flatten, DebugLayer, View


class SVHN(nn.Module):

    def __init__(self, num_domain):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5), # 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)), # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5), # 9
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)), # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2), # 4
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            Flatten(),
        )

        self.decoder_arr = nn.ModuleList([nn.Sequential(
            View([-1, 128, 4, 4]),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2),
        ) for _ in range(num_domain)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=3072),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3072, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=2),
        )


class SVHN_large(nn.Module):

    def __init__(self, num_domain):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5), # 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)), # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5), # 9
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)), # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2), # 4
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            Flatten(),
        )

        self.decoder_arr = nn.ModuleList([nn.Sequential(
            View([-1, 128, 4, 4]),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2),
        ) for _ in range(num_domain)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=3072),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3072, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=2),
        )