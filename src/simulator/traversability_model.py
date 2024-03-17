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
        self._risks = (
            self._infer_risk_map() if self._model_config.mode == "inference" else None
        )

    def _infer_risk_map(self, num_samples: int = 1000) -> torch.Tensor:
        """
        Precompute the risk map for inference mode using the predicted slip distribution.

        Parameters:
        - num_samples (int): Number of samples to draw from the predicted slip distribution.

        Returns:
        - risks (torch.Tensor): Risk map for inference mode.
        """
        distributions = self._grid_map.distributions["predictions"]
        if self._model_config.inference_metric == "expected_value":
            return distributions.mean
        else:
            samples = distributions.sample((num_samples,))
            var = torch.quantile(samples, self._model_config.confidence_value, dim=0)
            if self._model_config.inference_metric == "var":
                return var
            elif self._model_config.inference_metric == "cvar":
                tail_samples = samples[samples > var]
                return tail_samples.mean()

    def get_traversability(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get traversability at the given position.
        Observation mode: Draw samples from the slip distribution during trajectory execution.
        Inference mode: Infer traversability from the predicted slip distribution during trajectory planning.

        Parameters:
        - states (torch.Tensor): States of the robot as batch of position tensors shaped [batch_size, num_positions, 3].

        Returns:
        - trav (torch.Tensor): Traversability at the given position.
        """
        if self._model_config.mode == "observation":
            distributions = self._grid_map.get_values_at_positions(
                self._grid_map.distributions["latent_models"], states
            )
            return 1 - torch.clamp(distributions.sample(), 0, 1)
        elif self._model_config.mode == "inference":
            risks = self._grid_map.get_values_at_positions(self._risks, states)
            return 1 - torch.clamp(risks, 0, 1)
