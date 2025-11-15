import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from baseline_cnn_mnist.model import MNIST_CNN
from baseline_cnn_mnist.model_cuda import MNIST_CNN_CUDA
from baseline_cnn_mnist.model_triton import MNIST_CNN_TRITON

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure_inference(model, dataloader):
    model.eval()
    total_time = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)

            torch.cuda.synchronize()
            start = time.time()

            _ = model(images)

            torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            total_batches += 1

    avg_time_ms = (total_time / total_batches) * 1000
    return avg_time_ms


def get_mnist_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transform,
        download=True,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    print("\n=========== MNIST CNN Inference Benchmark (PyTorch vs CUDA vs Triton) ===========\n")

    # Load shared pretrained weights
    weight_path = "baseline_cnn_mnist/baseline_cnn_mnist.pth"
    state_dict = torch.load(weight_path, map_location=DEVICE)
    print("Loaded shared model weights.\n")

    # 3 Versions of the CNN
    base_model = MNIST_CNN().to(DEVICE)
    base_model.load_state_dict(state_dict)

    cuda_model = MNIST_CNN_CUDA().to(DEVICE)
    cuda_model.load_state_dict(state_dict)

    triton_model = MNIST_CNN_TRITON().to(DEVICE)
    triton_model.load_state_dict(state_dict)

    batch_sizes = [16, 32, 64, 128]

    for bs in batch_sizes:
        loader = get_mnist_loader(bs)

        t_base = measure_inference(base_model, loader)
        t_cuda = measure_inference(cuda_model, loader)
        t_triton = measure_inference(triton_model, loader)

        print(f"Batch Size {bs}:")
        print(f"  PyTorch CNN   : {t_base:.4f} ms / batch")
        print(f"  CUDA CNN      : {t_cuda:.4f} ms / batch")
        print(f"  Triton CNN    : {t_triton:.4f} ms / batch\n")
    print("Benchmarking completed.\n")