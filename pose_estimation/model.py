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
            nn.Conv2d(in_channels = 2, out_channels = 48, kernel_size = 11, stride = 4, padding = 2), # Out : n x 48 x 55 x 55
            nn.BatchNorm2d(48), # Batch Normalization
            nn.ReLU(), # Activation
            nn.Dropout2d(p = 0.25), # Dropout
            # Max Pooling, always s = 2, z = 3
            # nn.MaxPool2d(kernel_size = 3, stride = 2), # Out : n x 48 x 27 x 27

            # Layer-2
            nn.Conv2d(in_channels = 48, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), # Out : n x 128 x 27 x 27
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Dropout2d(p = 0.25),

            # Layer-3
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout2d(p = 0.25),

            # Layer-4
            nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout2d(p = 0.25),

            # Layer-5
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p = 0.25),

            # Layer-6
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout2d(p = 0.25),

            # Layer-7
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size = 3, stride = 2),
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

        # RNN
        self.Recurrent = nn.LSTM(input_size = 128 * 6 * 6, hidden_size = 1024, num_layers = 2, batch_first = True)

        # Final linear
        self.Linear = nn.Linear(1024, 6)

    def forward(self, x):

        # x: (batch, seq_len, channel, width, height)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # CNN
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))

        x = self.Convolution(x)
        x = self.avgpool(x)

        # Fully connected
        # output = self.FullConnected(x)

        # RNN
        x = x.view(batch_size, seq_len, -1)
        output, _ = self.Recurrent(x)
        output = self.Linear(output)

        # Reshape to (batch * seq_len, 6)
        return output
    
    def predict(self, x):

        # x: (seq_len, channel, width, height)
        seq_len = x.size(0)

        # CNN
        x = x.view(seq_len, x.size(1), x.size(2), x.size(3))

        x = self.Convolution(x)
        x = self.avgpool(x)

        # RNN
        x = x.view(1, seq_len, -1)
        output, _ = self.Recurrent(x)
        output = self.Linear(output)

        # Reshape to (seq_len, 6)
        # output = output.view(seq_len, 6)

        return output
                
# Test the model
if __name__ == '__main__':
    model = PoseNet()
    model = model.cuda()
    
    # Pytorch model summary
    summary(model, input_size=(1, 5, 2, 600, 150))
    
    # Input as a tensor
    input_ = torch.randn(2, 7, 2, 600, 150)
    input_ = input_.cuda()
    
    output = model(input_)
    print(output.size())