import torch
import torch.nn as nn

class ManTraNet(nn.Module):
    def __init__(self):
        super(ManTraNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)
