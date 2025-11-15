import torch
import time

def measure_inference_speed(model, images):
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()

        out = model(images)

        torch.cuda.synchronize()
        end = time.time()

    return (end - start) * 1000  # ms
