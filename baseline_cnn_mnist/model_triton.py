import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline_cnn_mnist.activations_triton import triton_swish_activation, triton_gelu_activation

class MNIST_CNN_TRITON(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = triton_swish_activation(self.conv1(x))
        x = self.pool(x)

        x = triton_swish_activation(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = triton_gelu_activation(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
