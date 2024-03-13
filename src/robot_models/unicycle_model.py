"""
Masafumi Endo, 2024.
"""

from __future__ import annotations
from typing import Optional

import torch

from src.environments.grid_map import GridMap
from src.robot_models.utils import ModelConfig


class UnicycleModel:
    def __init__(
        self,
        grid_map: GridMap,
        config: ModelConfig,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the unicycle model.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensor_data.
        - config (ModelConfig): Configuration for the model including mode, inference_metric, and confidence_value.
        - delta_t (float): Time step for simulation [s].
        - dtype (torch.dtype): Data type for torch tensors.
        - device (str): Device to run the model on.
        """
        self._grid_map = grid_map
        self._config = config
        self._dtype = dtype
        self._device = device

        # Define action space bounds
        self._min_action, self._max_action = self._define_action_space_bounds()

    def _define_action_space_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the minimum and maximum bounds for the action space (linear velocity, [m/s] and angular velocity, [rad/s]).
        
        Returns:
        - min_action (torch.Tensor): Minimum action bounds.
        - max_action (torch.Tensor): Maximum action bounds.
        """
        return (
            torch.tensor([0.0, -1.0], dtype=self._dtype, device=self._device),
            torch.tensor([1.0, 1.0], dtype=self._dtype, device=self._device),
        )

    def transit(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float
    ) -> torch.Tensor:
        """
        Compute the dynamics of the robot.

        Parameters:
        - state (torch.Tensor): Current state of the robot as batch of tensors (x, y, theta).
        - action (torch.Tensor): Control input to the robot as batch of tensors (linear velocity, [m/s] and angular velocity, [rad/s]).
        - delta_t (float): Time step for simulation [s].

        Returns:
        - next_state (torch.Tensor): Next state of the robot as batch of tensors (x, y, theta).
        """
        # Unpack state and action
        x, y, theta = state.unbind(1)
        v, omega = action.unbind(1)

        # Clamp action to the action space bounds
        v = torch.clamp(v, self._min_action[0], self._max_action[0])
        omega = torch.clamp(omega, self._min_action[1], self._max_action[1])

        # Get traversability at the current position
        trav = self._get_traversability(x, y)

        # Compute next state
        x += trav * v * torch.cos(theta) * delta_t
        y += trav * v * torch.sin(theta) * delta_t
        theta += trav * omega * delta_t
        # Wrap angle to [-pi, pi]
        theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi

        # Clamp next state to the grid map bounds
        x = torch.clamp(x, self._grid_map.x_limits[0], self._grid_map.x_limits[1])
        y = torch.clamp(y, self._grid_map.y_limits[0], self._grid_map.y_limits[1])
        return torch.stack([x, y, theta], dim=1)

    def _get_traversability(
        self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Get traversability at the given position.

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
        if self._config.mode == "observation":
            return self._observation_mode(x_index, y_index, n_samples)
        elif self._config.mode == "inference":
            return self._inference_mode(x_index, y_index)

    def _observation_mode(self, x_index: int, y_index: int, n_samples: int) -> torch.Tensor:
        """
        Get traversability at the given position in observation mode.

        Parameters:
        - x_index (int): X index in the grid map.
        - y_index (int): Y index in the grid map.
        - n_samples (int): Number of samples to draw from the slip distribution.

        Returns:
        - trav (torch.Tensor): Traversability at the given position.
        """
        distribution = self._grid_map.tensor_data["slips"]
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
        distribution = self._grid_map.tensor_data["predictions"]
        if self._config.inference_metric == "expected_value":
            # Compute expected value
            inference = distribution.mean[y_index, x_index]
        elif self._config.inference_metric == "var":
            # Retrieve mean and standard deviation
            pred_mean = distribution.mean[y_index, x_index]
            pred_stddev = distribution.stddev[y_index, x_index]
            # Compute value at risk
            inference = pred_mean + pred_stddev * distribution.icdf(
                torch.tensor(self._config.confidence_value)
            )
        return 1 - torch.clamp(inference, 0, 1)