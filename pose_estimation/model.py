import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Simple AlexNet-like CNN model
# For one image, output 6 DOF pose
class PoseNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolution network
        # Input : n x 3 x 224 x 224
        self.Convolution = nn.Sequential(

            # Layer-1
            nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size = 11, stride = 4, padding = 2), # Out : n x 48 x 55 x 55
            nn.ReLU(), # Activation
            # Max Pooling, always s = 2, z = 3
            nn.MaxPool2d(kernel_size = 3, stride = 2), # Out : n x 48 x 27 x 27

            # Layer-2
            nn.Conv2d(in_channels = 48, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), # Out : n x 128 x 27 x 27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            # Layer-3
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # Layer-4
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.FullConnected = nn.Sequential(

            # Layer-1
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(),

            # Layer-2
            nn.Linear(2048, 2048),
            nn.ReLU(),

            # Layer-3
            nn.Linear(2048, 6)
        )

    def forward(self, x):
        x = self.Convolution(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 128 * 6 * 6)

        output = self.FullConnected(x)
        return output