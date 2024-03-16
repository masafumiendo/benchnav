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

    def compute_traversability(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute traversability at the given position.
        Here, 'computation' express retrieving traversability either from the observation or inference mode.
        Observation mode: Draw samples from the slip distribution during trajectory execution.
        Inference mode: Infer traversability from the predicted slip distribution during trajectory planning.

        Parameters:
        - states (torch.Tensor): States of the robot as batch of position tensors shaped [batch_size, num_positions, 3].

        Returns:
        - trav (torch.Tensor): Traversability at the given position.
        """
        if self._model_config.mode == "observation":
            distributions = self._grid_map.get_values_from_positions(
                states, "latent_models"
            )
            return 1 - torch.clamp(distributions.sample(), 0, 1)
        elif self._model_config.mode == "inference":
            raise NotImplementedError
