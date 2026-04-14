import os
import torch

if os.getenv("DEVICE") == "mps":
    torchArangeFloat = torch.float32
else:
    torchArangeFloat = torch.float64