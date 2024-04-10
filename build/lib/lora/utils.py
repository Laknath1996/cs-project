import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a

    Args:
        seed (int): The desired seed.
    """

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)