import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from baseline_cnn_mnist.losses_pytorch import focal_loss_pytorch
from baseline_cnn_mnist.losses_triton import focal_loss_triton

torch.manual_seed(42)
log_probs = F.log_softmax(torch.randn(32, 10, device="cuda"), dim=1)
targets = torch.randint(0, 10, (32,), device="cuda")

loss_pt = focal_loss_pytorch(log_probs, targets)
loss_tr = focal_loss_triton(log_probs, targets)

print(f"PyTorch: {loss_pt.item():.6f}")
print(f"Triton:  {loss_tr.item():.6f}")
print(f"Diff:    {abs(loss_pt.item() - loss_tr.item()):.8f}")
print("✅ PASSED" if abs(loss_pt.item() - loss_tr.item()) < 1e-4 else "❌ FAILED")