import os
import torch
import numpy as np

def seed_everything(seed=42):
    """Seed everything for reproducibility,
    for numpy, torch, cuda
    Args:
        seed: int
    Return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
