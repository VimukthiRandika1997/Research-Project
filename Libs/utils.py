import os
import torch
import numpy as np


# Classes
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Functions
def create_early_stopper(patience, min_delta):
    """Create a early stopper for the training task
    Args:
        patience: number of epochs to be waited
        min_delta: minimum distance between two consecutive loss values
    """
    return EarlyStopper(patience, min_delta)


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
