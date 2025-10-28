import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from datetime import datetime

try:
    # Prefer relative import when running as a package
    from .model_baseline import BaselineCNN
except Exception:
    # Fallback to absolute import when running the script directly
    from baseline_cnn_mnist.model_baseline import BaselineCNN

from baseline_cnn_mnist.utils import set_seed, device_string, peak_gpu_mem_mb, save_json, accuracy


def get_loaders(batch_size: int, num_workers: int):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader, len(train_ds), len(test_ds)


def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            total_correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * total_correct / total


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    epoch_loss = 0.0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * y.size(0)
        total_seen += y.size(0)

    return epoch_loss / total_seen


def main():
    parser = argparse.ArgumentParser(description="Baseline CNN on MNIST with metrics")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (CUDA only)")
    # parser.add_argument("--out", type=str, default="profiling/baseline_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if (args.fp16 and torch.cuda.is_available()) else None

    train_loader, test_loader, n_train, n_test = get_loaders(args.batch_size, args.num_workers)

    # Warm up cuDNN/autotune, reset peak memory just before timing
    if torch.cuda.is_available():
        _x, _y = next(iter(train_loader))
        _x = _x.to(device)
        model(_x)
        torch.cuda.synchronize()

    epoch_times = []
    best_acc = 0.0

    # Reset peak memory counter before official training
    peak_mem = peak_gpu_mem_mb(reset=True)

    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        epoch_time = t1 - t0
        epoch_times.append(epoch_time)
        throughput = n_train / epoch_time

        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)

        # Update peak memory after each epoch
        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_mem = max(peak_mem, round(current_peak, 2))

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={test_acc:.2f}% | time={epoch_time:.2f}s | "
              f"throughput={throughput:.0f} img/s")

    # -------- SAVE MODEL --------
    save_dir = "baseline_cnn_mnist/saved_model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "baseline_cnn_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Model saved at: {save_path}")

    # -------- SAVE METRICS JSON --------
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_throughput = n_train / avg_epoch_time

    metrics = {
        "device": device_string(),
        "dataset": "MNIST",
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "fp16": bool(scaler is not None),
        "best_accuracy_pct": round(best_acc, 2),
        "avg_epoch_time_sec": round(avg_epoch_time, 2),
        "avg_throughput_images_per_sec": round(avg_throughput, 2),
        "peak_gpu_memory_MB": round(peak_mem, 2),
        "model_path": save_path
    }

    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    exp_name = f"{timestamp}_baseline.json"  # you can make it dynamic for each model type

    report_dir = "report/runs"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, exp_name)

    # Save both to profiling and report folders
    save_json(metrics, args.out)
    save_json(metrics, report_path)

    print(f"✅ Training complete! Metrics saved to:")
    print(f" - Profiling: {args.out}")
    print(f" - Report: {report_path}")



if __name__ == "__main__":
    main()
