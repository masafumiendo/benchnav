"""
Masafumi Endo, 2024.
"""

from __future__ import annotations
from typing import Optional

import torch

from src.environments.grid_map import GridMap
from src.simulator.traversability_model import TraversabilityModel
from src.simulator.utils import ModelConfig


class UnicycleModel:
    def __init__(
        self,
        grid_map: GridMap,
        model_config: ModelConfig,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the unicycle model.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - model_config (ModelConfig): Configuration for the traversability model.
        - delta_t (float): Time step for simulation [s].
        - dtype (torch.dtype): Data type for torch tensors.
        - device (str): Device to run the model on.
        """
        self._grid_map = grid_map
        self._dtype = dtype
        self._device = device

        # Define traversability model
        self._traversability_model = TraversabilityModel(self._grid_map, model_config)

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
        trav = self._traversability_model.compute_traversability(x, y)

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