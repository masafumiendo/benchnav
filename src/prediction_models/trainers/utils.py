"""
author: Masafumi Endo
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.nn import Module


@dataclass
class ParamsModelTraining:
    """
    A structure containing parameters for model training, specifying hyperparameters, optimization settings, and other training-related parameters.

    Attributes:
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.
    - weight_decay (float): Weight decay for optimization.
    - num_epochs (int): Number of epochs for training.
    - num_iterations (int): Number of iterations for training.
    - save_interval (int): Interval for saving model checkpoints.
    - device (str): Device for training, either "cpu" or "cuda".
    """

    batch_size: Optional[int] = field(default=None)
    learning_rate: Optional[float] = field(default=None)
    weight_decay: Optional[float] = field(default=None)
    num_epochs: Optional[int] = field(default=None)
    num_iterations: Optional[int] = field(default=None)
    save_interval: Optional[int] = field(default=None)
    device: Optional[str] = field(default=None)

    def __post_init__(self):
        # Validate device
        if self.device not in ["cpu", "cuda"] and not self.device.startswith("cuda:"):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'.")


def load_model_state_dict(model: Module, model_directory: str, device: str) -> Module:
    """
    Load a trained model from a file.

    Parameters:
    - model (Module): The model need to be loaded.
    - model_directory (str): Directory containing the trained model.
    - device (str): Device for loading the model, either "cpu" or "cuda".

    Returns:
    - Module: The trained model.
    """
    if device not in ["cpu", "cuda"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'.")

    model.load_state_dict(torch.load(model_directory, map_location=device))
    model.eval()
    return model
