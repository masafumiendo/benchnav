"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, Optional, Union
import torch
import gpytorch
from gpytorch import Module
import pickle

from src.environments.slip_model import SlipModel
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.prediction_models.slip_regressors.gpr import GPModel
from src.utils.utils import set_randomness


class RegressorTrainer:
    def __init__(
        self,
        device: Optional[str],
        model_directory: str,
        data_directory: str,
        num_terrain_classes: int,
        slip_sensitivity_minmax: Tuple[float, float],
        slip_nonlinearity_minmax: Tuple[float, float],
        slip_offset_minmax: Tuple[float, float],
        noise_scale_minmax: Tuple[float, float],
        params_model_training: ParamsModelTraining,
    ):
        """
        Initialize the regressor trainer for the all possible latent slip functions.

        Parameters:
        - device (str): the device to use for training
        - model_directory (str): the directory containing the model file. No need to specify training parameters.
        - data_directory (str): the directory containing the data files
        - num_terrain_classes (int): the number of terrain classes
        - slip_sensitivity_minmax (Tuple[float, float]): the min and max value of the slip sensitivity.
        - slip_nonlinearity_minmax (Tuple[float, float]): the min and max value of the slip nonlinearity.
        - slip_offset_minmax (Tuple[float, float]): the min and max value of the slip offset.
        - noise_scale_minmax (Tuple[float, float]): the min and max value of the noise scale.
        - params_model_training (ParamsModelTraining): the parameters for model training
        """
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        # Set the number of terrain classes
        self.num_terrain_classes = num_terrain_classes

        # Validate the minmax values
        self.slip_sensitivity_minmax = self.validate_minmax(slip_sensitivity_minmax)
        self.slip_nonlinearity_minmax = self.validate_minmax(slip_nonlinearity_minmax)
        self.slip_offset_minmax = self.validate_minmax(slip_offset_minmax)
        self.noise_scale_minmax = self.validate_minmax(noise_scale_minmax)

        # Set the model training parameters
        self.params_model_training = params_model_training

        # Create the model directory if it does not exist
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        self.model_directory = os.path.join(
            model_directory,
            f"lr{self.params_model_training.learning_rate:.0e}_"
            f"iters{self.params_model_training.num_iterations:03d}",
        )
        self.learned_models_directory = os.path.join(self.model_directory, "models")
        if not os.path.exists(self.learned_models_directory):
            os.makedirs(self.learned_models_directory)

        # Create the data directory if it does not exist
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        self.data_directory = os.path.join(data_directory, "slip_models")
        if not os.path.exists(self.data_directory):
            os.makedirs(os.path.join(self.data_directory, "models"))
            os.makedirs(os.path.join(self.data_directory, "observations"))

    def validate_minmax(self, minmax: Tuple[float, float]) -> Tuple[float, float]:
        """
        Validate the minmax tuple.

        Parameters:
        - minmax (Tuple[float, float]): the minmax tuple to validate

        Returns:
        - minmax (Tuple[float, float]): the validated minmax tuple
        """
        if minmax[0] >= minmax[1]:
            raise ValueError("The minimum value must be less than the maximum value.")
        return minmax

    def train_all_models(self) -> None:
        """
        Train the regressor model for all possible latent slip functions.
        """
        for terrain_class in range(self.num_terrain_classes):
            self.train(terrain_class)

    def train(self, terrain_class: int) -> Module:
        """
        Train the regressor model for the given terrain class.

        Parameters:
        - terrain_class (int): the terrain class used as a seed for the random number generator.

        Returns:
        - model (Module): the trained regressor model
        """
        # Initialize the latent slip model with the given terrain class
        slip_sensitivity, slip_nonlinearity, slip_offset, noise_scale = self.set_slip_model_parameters(
            terrain_class
        )
        slip_model = SlipModel(
            slip_sensitivity=slip_sensitivity,
            slip_nonlinearity=slip_nonlinearity,
            slip_offset=slip_offset,
            base_noise_scale=noise_scale,
            seed=terrain_class,
            device=self.device,
        )
        phis = torch.linspace(-45, 45, 2500).to(self.device)
        observed_slips = slip_model.observe_slip(phis)

        # Initialize the GPModel
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPModel(phis, observed_slips, likelihood).to(self.device)

        # Set the model and likelihood to training mode
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.params_model_training.learning_rate
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.params_model_training.num_iterations):
            optimizer.zero_grad()
            output = model(phis)
            loss = -mll(output, observed_slips)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    f"Iteration {i+1}/{self.params_model_training.num_iterations} - Loss: {loss.item()}"
                )

        # Save the actual and learned slip models
        self.save(slip_model, phis, observed_slips, model, terrain_class)

    def set_slip_model_parameters(
        self, terrain_class: int
    ) -> Tuple[float, float, float, float]:
        """
        Set the slip model parameters for the given terrain class.

        Parameters:
        - terrain_class (int): the terrain class used as a seed for the random number generator.

        Returns:
        - slip_sensitivity (float): the slip sensitivity for the given terrain class
        - slip_nonlinearity (float): the slip nonlinearity for the given terrain class
        - slip_offset (float): the slip offset for the given terrain class
        - noise_scale (float): the noise scale for the given terrain class
        """
        set_randomness(terrain_class)
        slip_sensitivity = self.uniform_sampling(
            self.slip_sensitivity_minmax[0], self.slip_sensitivity_minmax[1]
        )
        slip_nonlinearity = self.uniform_sampling(
            self.slip_nonlinearity_minmax[0], self.slip_nonlinearity_minmax[1]
        )
        slip_offset = self.uniform_sampling(
            self.slip_offset_minmax[0], self.slip_offset_minmax[1]
        )
        noise_scale = self.uniform_sampling(
            self.noise_scale_minmax[0], self.noise_scale_minmax[1]
        )
        return slip_sensitivity, slip_nonlinearity, slip_offset, noise_scale

    def uniform_sampling(
        self, min_val: float, max_val: float, num_samples: int = 1
    ) -> Union[float, torch.Tensor]:
        """
        Uniformly sample the given range.

        Parameters:
        - min_val (float): the minimum value of the range
        - max_val (float): the maximum value of the range
        - num_samples (int): the number of samples

        Returns:
        - samples (Union[float, torch.Tensor]): the uniformly sampled values
        """
        samples = torch.rand(num_samples) * (max_val - min_val) + min_val
        return samples if num_samples > 1 else samples.item()

    def save(
        self,
        slip_model: SlipModel,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        model: Module,
        terrain_class: int,
    ) -> None:
        """
        Save the actual and learned slip models.

        Parameters:
        - slip_model (SlipModel): the actual slip model
        - train_x (torch.Tensor): the training inputs
        - train_y (torch.Tensor): the training outputs
        - model (Module): the learned regressor model
        - terrain_class (int): the terrain class
        """
        # Save the actual slip model
        with open(
            os.path.join(self.data_directory, f"models/{terrain_class:02d}_class.pkl"),
            "wb",
        ) as f:
            pickle.dump(slip_model, f)

        # Save the training inputs and outputs as a dictionary
        torch.save(
            {"train_x": train_x, "train_y": train_y},
            os.path.join(
                self.data_directory, f"observations/{terrain_class:02d}_class.pth"
            ),
        )

        # Save the learned regressor model
        torch.save(
            model.state_dict(),
            os.path.join(
                self.learned_models_directory, f"{terrain_class:02d}_class.pth"
            ),
        )
