# baseline_cnn/utils.py
import torch
import os

def save_model(model, filename):
    os.makedirs("baseline_cnn/saved_models", exist_ok=True)
    path = os.path.join("baseline_cnn/saved_models", filename)
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")

def load_model(model, filename):
    path = os.path.join("baseline_cnn/saved_models", filename)
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from: {path}")
    return model
