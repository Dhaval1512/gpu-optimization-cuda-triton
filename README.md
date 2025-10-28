# 🧠 gpu-optimization-cuda-triton
**Confidential Research Project**  
Comparing CUDA and OpenAI Triton implementations for CNN optimization and kernel fusion.

---

## ⚙️ Phase 1 — Baseline CNN (MNIST)

### 🎯 Objective
This phase establishes the **baseline performance** of a clean CNN trained on the **MNIST** dataset using standard PyTorch operators.  
All later phases (kernel fusion, Triton implementations, mixed-precision, etc.) will benchmark their improvements against these baseline metrics.

---

### 🚀 Quick Start
```bash
# (optional) create & activate a virtual environment
python3 -m venv gpu_env && source gpu_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio pandas openpyxl --index-url https://download.pytorch.org/whl/cu121

# run baseline training
bash run_train.sh
# or equivalently
python3 -m baseline_cnn_mnist.train --epochs 10 --batch-size 128 --lr 1e-3 --num-workers 4 --exp-name baseline
