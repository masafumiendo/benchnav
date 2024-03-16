"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch

from src.environments.grid_map import GridMap


class Objectives:
    def __init__(self, grid_map: GridMap, goal_pos: torch.Tensor) -> None:
        """
        Initialize the traversability model.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - goal_pos (torch.Tensor): Goal position of the robot for the stage cost.
        """
        self._grid_map = grid_map
        self._goal_pos = goal_pos

    def stage_cost(
        self, state: torch.Tensor, action: torch.Tensor, stuck_threshold: float = 0.9
    ) -> torch.Tensor:
        """
        Compute stage cost at the given position.

        Parameters:
        - state (torch.Tensor): State of the robot as batch of position tensors shaped [batch_size, 3].
        - action (torch.Tensor): Action of the robot as batch of control tensors shaped [batch_size, 2].
        - stuck_threshold (float): Threshold for the robot to be considered stuck.

        Returns:
        - cost (torch.Tensor): Stage cost at the given position as batch of cost tensors shaped [batch_size].
        """
        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)

        distributions = self._grid_map.get_values_from_positions(
            state.unsqueeze(1), "latent_models"
        )
        slips = torch.clamp(distributions.mean, 0, 1)
        collision_cost = (slips > stuck_threshold).squeeze(1)

        return goal_cost + 1e4 * collision_cost

    def terminal_cost(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute terminal cost at the given position.

        Parameters:
        - state (torch.Tensor): State of the robot as batch of position tensors shaped [batch_size, 3].

        Returns:
        - cost (torch.Tensor): Terminal cost at the given position as batch of cost tensors shaped [batch_size].
        """
        return self.stage_cost(state, torch.zeros_like(state[:, :2]))
