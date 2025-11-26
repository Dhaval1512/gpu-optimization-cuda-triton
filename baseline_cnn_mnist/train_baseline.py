import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from baseline_cnn_mnist.model_pytorch import MNIST_CNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(num_epochs=1, batch_size=64):
    # MNIST transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MNIST_CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.NLLLoss()

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), "baseline_cnn_mnist.pth")
    print("Model saved as baseline_cnn_mnist.pth")

    return model

if __name__ == "__main__":
    train_model(num_epochs=1)
