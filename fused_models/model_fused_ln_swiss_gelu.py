#!/usr/bin/env python3
"""CNN model using Fusion 3: LayerNorm + GELU + Swish"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion3CNN(nn.Module):
    """
    CNN for MNIST demonstrating Fusion 3: LayerNorm + GELU + Swish
    
    This model applies the fused operation after convolutional layers,
    combining normalization and dual activation in a single kernel.
    """
    def __init__(self, use_fusion=False, backend='triton'):
        super(Fusion3CNN, self).__init__()
        self.use_fusion = use_fusion
        self.backend = backend
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # LayerNorm instead of BatchNorm to use fusion
        self.ln1 = nn.LayerNorm([32, 26, 26])
        self.ln2 = nn.LayerNorm([64, 12, 12])
        
        # Fully connected
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)
        
        # Import fusion operations if needed
        if use_fusion:
            if backend == 'triton':
                from triton_kernels.triton_ops import triton_fused_ln_gelu_swish
                self.fused_ln_gelu_swish = lambda x, g, b, eps: triton_fused_ln_gelu_swish(x, g, b, eps)
            elif backend == 'cuda':
                import cuda_ops
                # CUDA version takes eps as positional arg, not keyword
                self.fused_ln_gelu_swish = lambda x, g, b, eps: cuda_ops.cuda_fused_ln_gelu_swish(x, g, b, eps)
            else:
                raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)  # [B, 32, 26, 26]
        
        # Reshape for LayerNorm + GELU + Swish fusion
        B, C, H, W = x.shape
        x_flat = x.view(B, C * H * W)  # [B, 21632]
        
        if self.use_fusion:
            # Get LayerNorm parameters for this layer
            gamma = self.ln1.weight.view(-1)
            beta = self.ln1.bias.view(-1)
            
            x_flat = self.fused_ln_gelu_swish(x_flat, gamma, beta, 1e-5)
        else:
            # Unfused: LayerNorm + GELU + Swish
            x_flat = F.layer_norm(x_flat, (C * H * W,), 
                                 self.ln1.weight.view(-1),
                                 self.ln1.bias.view(-1))
            x_gelu = F.gelu(x_flat)
            x_swish = x_flat * torch.sigmoid(x_flat)
            x_flat = x_gelu + x_swish
        
        x = x_flat.view(B, C, H, W)
        x = F.max_pool2d(x, 2)  # [B, 32, 13, 13]
        
        # Conv block 2
        x = self.conv2(x)  # [B, 64, 11, 11]
        x = F.max_pool2d(x, 2)  # [B, 64, 5, 5]
        
        # Fully connected
        x = torch.flatten(x, 1)  # [B, 1600]
        x = self.fc1(x)  # [B, 128]
        
        # Apply Fusion 3 to FC layer
        if self.use_fusion:
            gamma = self.ln3.weight
            beta = self.ln3.bias
            x = self.fused_ln_gelu_swish(x, gamma, beta, 1e-5)
        else:
            x = F.layer_norm(x, (128,), self.ln3.weight, self.ln3.bias)
            x_gelu = F.gelu(x)
            x_swish = x * torch.sigmoid(x)
            x = x_gelu + x_swish
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing Fusion3CNN architecture...")
    
    model = Fusion3CNN(use_fusion=False).cuda()
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 1, 28, 28).cuda()
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    print("✅ Model test passed!")