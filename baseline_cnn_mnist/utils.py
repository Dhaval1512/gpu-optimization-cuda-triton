import os
import json
import math
import random
import torch
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Allow autotune for better performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def device_string():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"cuda:{name}"
    return "cpu"


def peak_gpu_mem_mb(reset: bool = True) -> float:
    if torch.cuda.is_available():
        if reset:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return round(peak, 2)
    return 0.0


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0
