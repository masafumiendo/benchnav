"""
Masafumi Endo, 2024.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple

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
        self._model_config = model_config
        self._traversability_model = TraversabilityModel(
            self._grid_map, self._model_config
        )

        # Define action space bounds
        self.min_action, self.max_action = self._define_action_space_bounds()

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
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the dynamics of the robot.

        Parameters:
        - state (torch.Tensor): Current state of the robot as batch of tensors (x, y, theta).
        - action (torch.Tensor): Control input to the robot as batch of tensors (linear velocity, [m/s] and angular velocity, [rad/s]).
        - delta_t (float): Time step for simulation [s].

        Returns:
        - next_state (torch.Tensor): Next state of the robot as batch of tensors (x, y, theta).
        - trav (torch.Tensor): Traversability at the next position if in observation mode.
        """
        # Get traversability at the current position
        trav = self.get_traversability(state.unsqueeze(1)).squeeze(1)

        # Unpack state and action
        x, y, theta = state.unbind(1)
        v, omega = action.unbind(1)

        # Clamp action to the action space bounds
        v = torch.clamp(v, self.min_action[0], self.max_action[0])
        omega = torch.clamp(omega, self.min_action[1], self.max_action[1])

        # Compute next state
        x += trav * v * torch.cos(theta) * delta_t
        y += trav * v * torch.sin(theta) * delta_t
        theta += trav * omega * delta_t
        # Wrap angle to [-pi, pi]
        theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi

        # Clamp next state to the grid map bounds
        x = torch.clamp(x, self._grid_map.x_limits[0], self._grid_map.x_limits[1])
        y = torch.clamp(y, self._grid_map.y_limits[0], self._grid_map.y_limits[1])
        next_state = torch.stack([x, y, theta], dim=1)
        # Return next state and traversability if in observation mode
        if self._model_config.mode == "observation":
            return next_state, trav
        elif self._model_config.mode == "inference":
            return next_state

    def get_traversability(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get traversability at the given position.

        Parameters:
        - states (torch.Tensor): State of the robot as batch of position tensors shaped [batch_size, num_positions, 3].

        Returns:
        - trav (torch.Tensor): Traversability at the given position as batch of traversability tensors shaped [batch_size, num_positions].
        """
        return self._traversability_model.get_traversability(states)
