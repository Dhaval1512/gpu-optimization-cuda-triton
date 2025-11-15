import torch

def focal_loss_pytorch(log_probs, targets, alpha=0.25, gamma=2.0):
    pt = torch.exp(log_probs)
    focal_weight = (1 - pt) ** gamma
    focal_loss = -alpha * focal_weight * log_probs
    return focal_loss.gather(1, targets.unsqueeze(1)).squeeze(1).mean()