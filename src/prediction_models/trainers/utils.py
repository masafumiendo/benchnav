"""
Masafumi Endo, 2024
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
from torch.nn import Module
from gpytorch import Module
from gpytorch.likelihoods import GaussianLikelihood


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


def load_slip_regressors(
    model: Module,
    num_terrain_classes: int,
    model_directory: str,
    train_data_directory: str,
    device: str,
) -> Dict[int, Module]:
    """
    Load trained slip regressors for each terrain class.

    Parameters:
    - model (Module): The model need to be loaded (e.g., ExactGPModel).
    - num_terrain_classes (int): Number of terrain classes.
    - model_directory (str): Directory containing the trained model.
    - train_data_directory (str): Directory containing the training data.
    - device (str): Device for loading the model, either "cpu" or "cuda".

    Returns:
    - Dict[int, Module]: A dictionary containing the trained slip regressors for each terrain class.
    """

    all_gp_models = {}  # Dictionary to store the GP model
    all_train_data = {}  # Dictionary to store the training data
    for i in range(num_terrain_classes):
        # Load the training data
        train_data = torch.load(
            os.path.join(train_data_directory, f"slip_observations/{i:02d}_class.pth")
        )
        train_x = train_data["train_x"].to(device)
        train_y = train_data["train_y"].to(device)
        # Initialize the GP model
        likelihood = GaussianLikelihood().to(device=device)
        gp_model = model(train_x, train_y, likelihood).to(device)
        # Load the trained model
        gp_model = load_model_state_dict(
            gp_model, os.path.join(model_directory, f"models/{i:02d}_class.pth"), device
        )
        # Store the model
        all_gp_models[i] = gp_model
        # Store the training data
        all_train_data[i] = train_data

    return all_gp_models
