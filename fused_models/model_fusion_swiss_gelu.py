import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedCNN(nn.Module):
    """
    Modified CNN for MNIST that uses fused GELU+Swish activation
    """
    def __init__(self, use_fusion=False, backend='triton'):
        super(FusedCNN, self).__init__()
        self.use_fusion = use_fusion
        self.backend = backend
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 64 * 12 * 12 = 9216
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
        # Import fusion operations if needed
        if use_fusion:
            if backend == 'triton':
                from triton_kernels.triton_ops import triton_fused_gelu_swish
                self.fused_activation = triton_fused_gelu_swish
            elif backend == 'cuda':
                import cuda_ops
                self.fused_activation = cuda_ops.cuda_fused_gelu_swish
            else:
                raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        
        # Apply GELU+Swish activation (fused or unfused)
        if self.use_fusion:
            x = self.fused_activation(x)
        else:
            # Unfused: compute GELU and Swish separately
            x_gelu = F.gelu(x)
            x_swish = x * torch.sigmoid(x)
            x = x_gelu + x_swish
        
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        x = self.conv2(x)
        
        # Apply GELU+Swish activation again
        if self.use_fusion:
            x = self.fused_activation(x)
        else:
            x_gelu = F.gelu(x)
            x_swish = x * torch.sigmoid(x)
            x = x_gelu + x_swish
        
        x = F.max_pool2d(x, 2)
        
        # Fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        # Apply GELU+Swish in FC layer
        if self.use_fusion:
            x = self.fused_activation(x)
        else:
            x_gelu = F.gelu(x)
            x_swish = x * torch.sigmoid(x)
            x = x_gelu + x_swish
        
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)