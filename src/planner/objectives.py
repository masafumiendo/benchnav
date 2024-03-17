"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch

from src.simulator.robot_model import UnicycleModel


class Objectives:
    def __init__(self, dynamics: UnicycleModel, goal_pos: torch.Tensor) -> None:
        """
        Initialize the objectives for the motion planning problem.

        Parameters:
        - dynamics (UnicycleModel): Dynamics model of the robot.
        - goal_pos (torch.Tensor): Goal position of the robot for the stage cost.
        """
        self._dynamics = dynamics
        self._goal_pos = goal_pos

    def stage_cost(
        self, state: torch.Tensor, action: torch.Tensor, stuck_threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Compute stage cost at the given position.

        Parameters:
        - state (torch.Tensor): State of the robot as batch of position tensors shaped [batch_size, 3].
        - action (torch.Tensor): Action of the robot as batch of control tensors shaped [batch_size, 2].
        - stuck_threshold (float): Threshold for the robot to be considered stuck (low traversability).

        Returns:
        - cost (torch.Tensor): Stage cost at the given position as batch of cost tensors shaped [batch_size].
        """
        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)

        # Compute collision cost against stuck situations
        trav = self._dynamics.get_traversability(state.unsqueeze(1)).squeeze(1)
        collision_cost = trav <= stuck_threshold

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
