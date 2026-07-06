import os
import torch


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


