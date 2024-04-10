"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from simulator.problem_formulation.robot_model import UnicycleModel
from simulator.problem_formulation.objectives import Objectives


class DWA(nn.Module):
    """
    Dynamic Window Approach, 
    J. Fox et al., IROS, 1997.
    """

    def __init__(
        self,
        horizon: int,
        dim_state: int,
        dim_control: int,
        dynamics: UnicycleModel,
        objectives: Objectives,
        a_lim: torch.Tensor,
        delta_t: float,
        lookahead_distance: float = 1.0,
        num_lin_vel: int = 10,
        num_ang_vel: int = 10,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        """
        Initialize the Dynamic Window Approach.

        Parameters:
        - horizon (int): Predictive horizon length.
        - dim_state (int): Dimension of state.
        - dim_control (int): Dimension of control.
        - dynamics (UnicycleModel): Dynamics model.
        - objectives (Objectives): Objectives class.
        - a_lim (torch.Tensor): Maximum acceleration.
        - delta_t (float): Time step for simulation [s].
        - lookahead_distance (float): Lookahead distance for the sub-goal selection.
        - num_lin_vel (int): Number of linear velocity samples.
        - num_ang_vel (int): Number of angular velocity samples.
        - device (torch.device): Device to run the solver.
        - dtype (torch.dtype): Data type to run the solver.
        - seed (int): Seed for torch.
        """
        super().__init__()

        # torch seed
        torch.manual_seed(seed)

        # check dimensions
        assert dynamics.min_action.shape == (
            dim_control,
        ), "minimum actions must be a tensor of shape (dim_control,)"
        assert dynamics.max_action.shape == (
            dim_control,
        ), "maximum actions must be a tensor of shape (dim_control,)"
        assert a_lim.shape == (
            dim_control,
        ), "acceleration limits must be a tensor of shape (dim_control,)"

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        self._horizon = horizon
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._stage_cost = objectives.stage_cost
        self._terminal_cost = objectives.terminal_cost
        self._u_min = (
            self._dynamics.min_action.clone().detach().to(self._device, self._dtype)
        )
        self._u_max = (
            self._dynamics.max_action.clone().detach().to(self._device, self._dtype)
        )
        self._a_lim = a_lim.clone().detach().to(self._device, self._dtype)
        self._delta_t = delta_t
        self._lookahead_distance = lookahead_distance
        self._num_lin_vel = num_lin_vel
        self._num_ang_vel = num_ang_vel

        self._previous_action_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )

        self._state_seq_batch = torch.zeros(
            self._num_lin_vel * self._num_ang_vel,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        self._weights = torch.zeros(
            self._num_lin_vel * self._num_ang_vel,
            device=self._device,
            dtype=self._dtype,
        )

        self.reference_path = None

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the optimal control input.

        Parameters:
        - state (torch.Tensor): Current state of the robot as a tensor (x, y, theta).

        Returns:
        - optimal_action_seq (torch.Tensor): Optimal action sequence.
        - optimal_state_seq (torch.Tensor): Optimal state sequence.
        """
        assert state.shape == (
            self._dim_state,
        ), "state must be a tensor of shape (dim_state,)"

        # Check input tensor is on the same device and data type
        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        if state.device != self._device:
            state = state.to(self._device, self._dtype)

        actions = self._generate_actions()

        # Simulate state sequences and compute costs
        state_seq_batch = self._simulate_state_sequences(state, actions)
        cost_batch = self._compute_costs(state_seq_batch, actions)

        # Find the optimal action and state sequences based on the minimum cost
        min_cost_index = torch.argmin(cost_batch, dim=0)
        optimal_action_seq = actions[min_cost_index].unsqueeze(0)
        optimal_state_seq = state_seq_batch[min_cost_index].unsqueeze(0)

        # Update previous action sequence
        self._previous_action_seq = optimal_action_seq
        self._state_seq_batch = state_seq_batch
        self._weights = torch.softmax(-cost_batch, dim=0)

        return optimal_action_seq, optimal_state_seq

    def update_reference_path(self, reference_path: torch.Tensor) -> None:
        """
        Update the reference path if available.

        Parameters:
        - reference_path (torch.Tensor): Reference path for the stage cost.
        """
        if reference_path is not None:
            assert (
                reference_path.shape[1] == 2
            ), "reference_path must be a tensor of shape (num_positions, 2)"
            self.reference_path = reference_path.to(self._device, self._dtype)

    def _generate_actions(self) -> torch.Tensor:
        """
        Generate set of actions within the dynamic window based on the current action and maximum acceleration.

        Parameters:

        Returns:
        - actions (torch.Tensor): Set of actions within the dynamic window.
        """
        # Adjust action limits based on the current action and maximum acceleration
        action = self._previous_action_seq[0, :]

        v_min = torch.max(self._u_min[0], action[0] - self._a_lim[0] * self._delta_t)
        v_max = torch.min(self._u_max[0], action[0] + self._a_lim[0] * self._delta_t)
        omega_min = torch.max(
            self._u_min[1], action[1] - self._a_lim[1] * self._delta_t
        )
        omega_max = torch.min(
            self._u_max[1], action[1] + self._a_lim[1] * self._delta_t
        )

        # Generate linear and angular velocities within the dynamic window
        vs = torch.linspace(
            v_min, v_max, self._num_lin_vel, device=self._device, dtype=self._dtype
        )
        omegas = torch.linspace(
            omega_min,
            omega_max,
            self._num_ang_vel,
            device=self._device,
            dtype=self._dtype,
        )
        actions = torch.cartesian_prod(vs, omegas)
        return actions

    def _simulate_state_sequences(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate the state sequence based on the current state and action.

        Parameters:
        - state (torch.Tensor): Current state of the robot as a tensor (x, y, theta).
        - actions (torch.Tensor): All possible actions as a batch of control tensors shaped [num_actions, 2].

        Returns:
        - state_seq_batch (torch.Tensor): State sequences.
        """
        state_seq_batch = torch.zeros(
            actions.shape[0],
            self._horizon + 1,
            3,
            device=self._device,
            dtype=self._dtype,
        )
        state_seq_batch[:, 0, :] = state.unsqueeze(0).repeat(actions.shape[0], 1)
        for t in range(self._horizon):
            state_seq_batch[:, t + 1, :] = self._dynamics.transit(
                state_seq_batch[:, t, :], actions
            )
        return state_seq_batch

    def _compute_costs(
        self, state_seq_batch: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cost based on the state sequence.

        Parameters:
        - state_seq_batch (torch.Tensor): Batch of state sequences.
        - actions (torch.Tensor): All possible actions as a batch of control tensors shaped [num_actions, horizon, 2].

        Returns:
        - cost_batch (torch.Tensor): Total costs.
        """
        sub_goal_pos = (
            self._select_sub_goal(state_seq_batch[0, 0, :])
            if self.reference_path is not None
            else None
        )
        cost_batch = torch.zeros(
            state_seq_batch.shape[0], device=self._device, dtype=self._dtype
        )
        for t in range(self._horizon):
            cost_batch += self._stage_cost(
                state_seq_batch[:, t, :], actions, sub_goal_pos
            )

        cost_batch += self._terminal_cost(state_seq_batch[:, -1, :])

        return cost_batch

    def _select_sub_goal(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select sub-goal position based on the reference path.

        Parameters:
        - state (torch.Tensor): State of the robot as a tensor (x, y, theta).

        Returns:
        - sub_goal_pos (torch.Tensor): Sub-goal position.
        """
        deltas = self.reference_path - state[:2]

        distances = torch.norm(deltas, dim=1)
        angles = torch.atan2(deltas[:, 1], deltas[:, 0]) - state[2]

        valid_mask = (angles.abs() < torch.pi / 2) & (
            distances > self._lookahead_distance
        )

        if valid_mask.any():
            min_distance_ahead = distances[valid_mask].min()
            sub_goal_index = torch.where(distances == min_distance_ahead)[0][0]
            sub_goal_pos = self.reference_path[sub_goal_index]
        else:
            sub_goal_pos = self.reference_path[-1]
        return sub_goal_pos

    def get_top_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top samples 

        Returns:
        - top_samples (torch.Tensor): Top samples.
        - top_weights (torch.Tensor): Top weights.
        """
        top_indices = torch.argsort(self._weights, descending=True)
        top_samples = self._state_seq_batch[top_indices]
        top_weights = self._weights[top_indices]

        return top_samples, top_weights
