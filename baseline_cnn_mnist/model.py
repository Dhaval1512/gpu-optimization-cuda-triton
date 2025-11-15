import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 input channel (grayscale), 16 output channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # After 2 pools: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv + ReLU + Pool

        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))               # FC + ReLU
        x = self.fc2(x)                       # Output layer

        return F.log_softmax(x, dim=1) # Log-Softmax for classification
