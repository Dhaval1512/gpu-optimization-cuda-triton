import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fused_models.model_fusion_swiss_gelu import FusedCNN, count_parameters

def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return avg_loss, accuracy

def test(model, device, test_loader):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def main():
    print("\n" + "="*80)
    print("  TRAINING FUSED CNN ON MNIST")
    print("="*80 + "\n")
    
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 5
    lr = 0.01
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create model (train without fusion for compatibility)
    model = FusedCNN(use_fusion=False).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}\n")
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    
    # Training loop
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        print(f"{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*80}")
        
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        scheduler.step()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'fused_cnn_mnist.pth')
            print(f"✅ Saved best model with accuracy: {test_acc:.2f}%\n")
    
    print("="*80)
    print(f"Training complete! Best test accuracy: {best_accuracy:.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()