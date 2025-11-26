import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from baseline_cnn_mnist.model_pytorch import MNIST_CNN
from baseline_cnn_mnist.losses_cuda import focal_loss_cuda
from baseline_cnn_mnist.losses_triton import focal_loss_triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(loss_type="pytorch", num_epochs=1, batch_size=64, alpha=0.25, gamma=2.0):
    """
    Train MNIST CNN with different loss implementations
    
    Args:
        loss_type: "pytorch", "cuda", or "triton"
        num_epochs: number of training epochs
        batch_size: training batch size
        alpha: focal loss alpha parameter
        gamma: focal loss gamma parameter
    """
    
    # MNIST transforms
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = MNIST_CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Select loss function
    if loss_type == "pytorch":
        criterion = torch.nn.NLLLoss()
        print("Using PyTorch NLLLoss")
    elif loss_type == "cuda":
        criterion = lambda output, target: focal_loss_cuda(output, target, alpha, gamma)
        print(f"Using CUDA Focal Loss (alpha={alpha}, gamma={gamma})")
    elif loss_type == "triton":
        criterion = lambda output, target: focal_loss_triton(output, target, alpha, gamma)
        print(f"Using Triton Focal Loss (alpha={alpha}, gamma={gamma})")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(images)  # log_softmax output
            
            if loss_type == "pytorch":
                loss = criterion(output, labels)
            else:
                loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss = {avg_loss:.4f}")
    
    # Save model
    save_path = f"baseline_cnn_mnist/cnn_focal_{loss_type}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="pytorch", 
                        choices=["pytorch", "cuda", "triton"],
                        help="Loss function implementation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    
    args = parser.parse_args()
    
    train_model(
        loss_type=args.loss,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma
    )