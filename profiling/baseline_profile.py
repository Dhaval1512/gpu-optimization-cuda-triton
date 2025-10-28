# profiling/baseline_profile.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time, json, os
from tqdm import tqdm
from baseline_cnn.baseline_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./baseline_cnn/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# --- Model, Loss, Optimizer ---
model = SimpleCNN().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Profiling metrics ---
num_epochs = 5
epoch_times = []
peak_memory = 0
total_images = len(trainloader.dataset)

for epoch in range(num_epochs):
    model.train()
    start_time = time.time()

    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track peak GPU memory
        torch.cuda.synchronize()
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    print(f"Epoch [{epoch+1}/{num_epochs}] Time: {epoch_time:.2f}s")

# --- Compute averages ---
avg_epoch_time = sum(epoch_times) / num_epochs
throughput = total_images / avg_epoch_time

metrics = {
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "num_epochs": num_epochs,
    "avg_epoch_time_sec": round(avg_epoch_time, 2),
    "avg_throughput_images_per_sec": round(throughput, 2),
    "peak_gpu_memory_MB": round(peak_memory / (1024 ** 2), 2)
}

# --- Save results ---
os.makedirs("profiling", exist_ok=True)
with open("profiling/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Profiling complete! Results saved to profiling/baseline_metrics.json")
print(json.dumps(metrics, indent=2))
