import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test configurations
configs = [
    {"name": "GELU", "size": (1024, 1024)},
    {"name": "LayerNorm", "size": (4096, 1024)},
    {"name": "Conv2D", "size": (32, 3, 224, 224)},
]

print("\n" + "="*70)
print("  GPU KERNEL PROFILING - PYTORCH OPERATIONS")
print("="*70)

for config in configs:
    print(f"\n{config['name']}:")
    
    if config['name'] == "GELU":
        x = torch.randn(config['size'], device=device)
        for i in range(50):
            torch.cuda.synchronize()
            y = F.gelu(x)
            torch.cuda.synchronize()
            if i % 10 == 0:
                print(f"  Batch {i+1}/50")
    
    elif config['name'] == "LayerNorm":
        x = torch.randn(config['size'], device=device)
        ln = nn.LayerNorm(config['size'][1]).to(device)
        for i in range(50):
            torch.cuda.synchronize()
            y = ln(x)
            torch.cuda.synchronize()
            if i % 10 == 0:
                print(f"  Batch {i+1}/50")
    
    elif config['name'] == "Conv2D":
        x = torch.randn(config['size'], device=device)
        conv = nn.Conv2d(3, 64, 3, padding=1).to(device)
        for i in range(50):
            torch.cuda.synchronize()
            y = conv(x)
            torch.cuda.synchronize()
            if i % 10 == 0:
                print(f"  Batch {i+1}/50")

print("\nProfiling complete!")