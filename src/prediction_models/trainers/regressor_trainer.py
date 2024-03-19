"""
Masafumi Endo, 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Tuple, Dict
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import gpytorch
from gpytorch import Module

from src.prediction_models.trainers.utils import ParamsModelTraining
from src.prediction_models.slip_regressors.gpr import GPModel
from src.data.terrain_dataset import SlipRegressionDataset as Dataset


class RegressorTrainer:
    def __init__(
        self,
        device: Optional[str],
        model_directory: str,
        data_directory: str,
        num_terrain_classes: int,
        params_model_training: ParamsModelTraining,
        train_dataset: Dataset,
    ):
        """
        Initialize the regressor trainer for the all possible latent slip functions.

        Parameters:
        - device (str): the device to use for training
        - model_directory (str): the directory containing the model file. No need to specify training parameters.
        - data_directory (str): the directory containing the data files
        - num_terrain_classes (int): the number of terrain classes
        - params_model_training (ParamsModelTraining): the parameters for model training
        - train_dataset (Dataset): the training dataset
        """
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Set the training dataset then initialize the data loader
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(
            train_dataset, batch_size=params_model_training.batch_size, shuffle=True
        )

        # Set the number of terrain classes
        self.num_terrain_classes = num_terrain_classes

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
        self.data_directory = os.path.join(data_directory, "slip_observations")
        if not os.path.exists(self.data_directory):
            os.makedirs(os.path.join(self.data_directory))

    def train_all_models(self) -> None:
        """
        Train the regression model for all possible latent slip functions.
        """
        phis_dict, slips_dict = self.load_data()
        for terrain_class in range(self.num_terrain_classes):
            self.train(
                terrain_class, phis_dict[terrain_class], slips_dict[terrain_class]
            )

    def load_data(
        self, num_samples: int = 9999
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Load the training data for the given terrain class.

        Parameters:
        - num_samples (int): the maximum number of samples to load per terrain class

        Returns:
        - A tuple of dictionaries containing the training inputs and outputs
        """
        phis_dict = defaultdict(list)
        slips_dict = defaultdict(list)
        data_count_per_class = defaultdict(
            int
        )  # To track the number of data points per class

        for slopes, slips, t_classes in self.train_loader:
            for t_class in range(self.num_terrain_classes):
                mask = t_classes == t_class
                if torch.any(mask):
                    phis_class = slopes[mask]
                    slips_class = slips[mask]

                    # Append the training inputs and outputs to the dictionary
                    phis_dict[t_class].append(phis_class)
                    slips_dict[t_class].append(slips_class)

                    data_count_per_class[t_class] += phis_class.shape[0]

        # Concatenate lists into tensors and limit to num_samples
        for t_class in range(self.num_terrain_classes):
            phis_dict[t_class] = torch.cat(phis_dict[t_class], dim=0)[:num_samples]
            slips_dict[t_class] = torch.cat(slips_dict[t_class], dim=0)[:num_samples]

        return phis_dict, slips_dict

    def train(
        self, terrain_class: int, phis: torch.Tensor, slips: torch.Tensor
    ) -> Module:
        """
        Train the regression model for the given terrain class.

        Parameters:
        - terrain_class (int): the terrain class used as a seed for the random number generator.
        - phis (torch.Tensor): the inputs for training
        - slips (torch.Tensor): the outputs for training

        Returns:
        - model (Module): the trained regression model
        """
        # Initialize the GPModel
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPModel(phis, slips, likelihood).to(self.device)

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
            loss = -mll(output, slips)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    f"Iteration {i+1}/{self.params_model_training.num_iterations} - Loss: {loss.item()}"
                )

        # Save the actual and learned slip models
        self.save(phis, slips, model, terrain_class)

    def save(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        model: Module,
        terrain_class: int,
    ) -> None:
        """
        Save the actual and learned slip models.

        Parameters:
        - train_x (torch.Tensor): the training inputs
        - train_y (torch.Tensor): the training outputs
        - model (Module): the learned regressor model
        - terrain_class (int): the terrain class
        """
        # Save the training inputs and outputs as a dictionary
        torch.save(
            {"train_x": train_x, "train_y": train_y},
            os.path.join(self.data_directory, f"{terrain_class:02d}_class.pth"),
        )

        # Save the learned regressor model
        torch.save(
            model.state_dict(),
            os.path.join(
                self.learned_models_directory, f"{terrain_class:02d}_class.pth"
            ),
        )
