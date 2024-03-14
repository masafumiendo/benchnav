"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch

from src.environments.grid_map import GridMap
from src.simulator.utils import ModelConfig


class TraversabilityModel:
    def __init__(self, grid_map: GridMap, model_config: ModelConfig) -> None:
        """
        Initialize the traversability model.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - model_config (ModelConfig): Configuration for the traversability model.
        """
        self._grid_map = grid_map
        self._model_config = model_config

    def compute_traversability(
        self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Compute traversability at the given position.
        Here, 'computation' express retrieving traversability either from the observation or inference mode.
        Observation mode: Draw samples from the slip distribution during trajectory execution.
        Inference mode: Infer traversability from the predicted slip distribution during trajectory planning.

        Parameters:
        - x (torch.Tensor): X position in meters.
        - y (torch.Tensor): Y position in meters.
        - n_samples (int): Number of samples to draw from the slip distribution.

        Returns:
        - trav (torch.Tensor): Traversability at the given position.
        """
        # Check x and y tensors has only a unique position
        if not (x == x[0]).all() or not (y == y[0]).all():
            raise ValueError("Multiple positions are not supported!")
        x_index, y_index = self._grid_map.get_grid_indices_from_position(x[0], y[0])

        # Get slip at the given position
        if self._model_config.mode == "observation":
            return self._observation_mode(x_index, y_index, n_samples)
        elif self._model_config.mode == "inference":
            return self._inference_mode(x_index, y_index)

    def _observation_mode(
        self, x_index: int, y_index: int, n_samples: int
    ) -> torch.Tensor:
        """
        Get traversability at the given position in observation mode.

        Parameters:
        - x_index (int): X index in the grid map.
        - y_index (int): Y index in the grid map.
        - n_samples (int): Number of samples to draw from the slip distribution.

        Returns:
        - trav (torch.Tensor): Traversability at the given position.
        """
        distribution = self._grid_map.tensors["latent_models"]
        observation = distribution.sample((n_samples,))[y_index, x_index]
        return 1 - torch.clamp(observation, 0, 1)

    def _inference_mode(self, x_index: int, y_index: int) -> torch.Tensor:
        """
        Get traversability at the given position in inference mode.

        Parameters:
        - x_index (int): X index in the grid map.
        - y_index (int): Y index in the grid map.

        Returns:
        - trav (torch.Tensor): Traversability at the given position.
        """
        distribution = self._grid_map.tensors["predictions"]
        if self._model_config.inference_metric == "expected_value":
            # Compute expected value
            inference = distribution.mean[y_index, x_index]
        elif self._model_config.inference_metric == "var":
            # Retrieve mean and standard deviation
            pred_mean = distribution.mean[y_index, x_index]
            pred_stddev = distribution.stddev[y_index, x_index]
            # Compute value at risk
            inference = pred_mean + pred_stddev * distribution.icdf(
                torch.tensor(self._model_config.confidence_value)
            )
        return 1 - torch.clamp(inference, 0, 1)
