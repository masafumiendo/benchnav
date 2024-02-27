"""
aurhor: Masafumi Endo
"""

import random
import numpy as np
import torch
import opensimplex as simplex


def set_randomness(seed: int) -> None:
    """
    Set the randomness of the environment and models using the specified seed.

    Parameters:
    - seed (int): The seed to use for randomness.
    """
    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for OpenSimplex
    simplex.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
