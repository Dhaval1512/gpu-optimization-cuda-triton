import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, csv, statistics as stats
import torch.nn.functional as F
from baseline_cnn_mnist.losses_pytorch import focal_loss_pytorch
from baseline_cnn_mnist.losses_triton import focal_loss_triton

def benchmark_loss(fn, *args, iters=100):
    torch.cuda.synchronize()
    for _ in range(10): fn(*args)  # warmup
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record(); fn(*args); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return stats.mean(times), stats.pstdev(times)

results = []
for bs in [16, 32, 64, 128, 256]:
    log_probs = F.log_softmax(torch.randn(bs, 10, device="cuda"), dim=1)
    targets = torch.randint(0, 10, (bs,), device="cuda")
    
    mean, std = benchmark_loss(lambda: F.nll_loss(log_probs, targets))
    print(f"BS {bs:3d} | NLL:     {mean:.4f}±{std:.4f} ms")
    results.append(("nll", "pytorch", bs, mean, std))
    
    mean, std = benchmark_loss(lambda: focal_loss_pytorch(log_probs, targets))
    print(f"        | Focal-PT: {mean:.4f}±{std:.4f} ms")
    results.append(("focal", "pytorch", bs, mean, std))
    
    mean, std = benchmark_loss(lambda: focal_loss_triton(log_probs, targets))
    print(f"        | Focal-TR: {mean:.4f}±{std:.4f} ms\n")
    results.append(("focal", "triton", bs, mean, std))

os.makedirs("report", exist_ok=True)
with open("report/loss_benchmarks.csv", "w") as f:
    csv.writer(f).writerows([["loss", "impl", "batch", "mean_ms", "std_ms"]] + results)
print("Saved to report/loss_benchmarks.csv")