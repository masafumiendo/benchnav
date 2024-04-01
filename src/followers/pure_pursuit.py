"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        delta_t: float,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the PID controller.

        Parameters:
        - kp (float): The proportional gain.
        - ki (float): The integral gain.
        - kd (float): The derivative gain.
        - delta_t (float): The time step.
        - device (torch.device): The device on which the controller is run.
        - dtype (torch.dtype): The data type of the tensors.
        """
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._delta_t = delta_t
        self._device = device
        self._dtype = dtype

        self._batch_size = None
        self._previous_error = None
        self._integral = None

    def reset(self, batch_size) -> None:
        """
        Reset the controller.

        Parameters:
        - batch_size (int): The batch size.
        """
        self._batch_size = batch_size
        self._previous_error = torch.zeros(
            self._batch_size, device=self._device, dtype=self._dtype
        )
        self._integral = torch.zeros(
            self._batch_size, device=self._device, dtype=self._dtype
        )

    def update(self, current_error: torch.Tensor) -> torch.Tensor:
        """
        Compute the control input for the robot.

        Parameters:
        - current_error (torch.Tensor): The error signal.

        Returns:
        - control (torch.Tensor): The control input for the robot.
        """
        if self._previous_error is None or self._integral is None:
            raise ValueError(
                "The controller must be reset before the first update call."
            )
        # Compute the proportional term
        proportional = current_error

        # Compute the integral term
        self._integral += current_error * self._delta_t

        # Compute the derivative term
        derivative = (current_error - self._previous_error) / self._delta_t

        # Compute the control input
        control = (
            self._kp * proportional + self._ki * self._integral + self._kd * derivative
        )

        # Update the previous error
        self._previous_error = current_error

        return control


class PurePursuit:
    def __init__(
        self,
        lookahead_distance: float,
        lin_controller: PIDController,
        ang_controller: PIDController,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the Pure Pursuit controller.

        Parameters:
        - lookahead_distance (float): The distance ahead of the robot to look for the target point.
        - lin_controller (PIDController): The PID controller for the linear velocity.
        - ang_controller (PIDController): The PID controller for the angular velocity.
        - device (torch.device): The device on which the controller is run.
        - dtype (torch.dtype): The data type of the tensors.
        """
        self._lookahead_distance = lookahead_distance
        self._lin_controller = lin_controller
        self._ang_controller = ang_controller
        self._device = device
        self._dtype = dtype

    def reset(self, batch_size: int) -> None:
        """
        Reset the controller.

        Parameters:
        - batch_size (int): The batch size.
        """
        self._lin_controller.reset(batch_size)
        self._ang_controller.reset(batch_size)

    def control(
        self, state_batch: torch.Tensor, reference_paths_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the control input for the robot.

        Parameters:
        - state_batch (torch.Tensor): The batch of current states of the robot.
        - reference_paths_batch (torch.Tensor): The batch of paths to follow.

        Returns:
        - action (torch.Tensor): The control input for the robot.
        """
        # Compute the target point
        target_points = self._compute_target_points(state_batch, reference_paths_batch)
        linear_errors, angular_errors = self._calculate_errors(
            state_batch, target_points
        )

        # Compute the control input
        vs = self._lin_controller.update(linear_errors)
        omegas = self._ang_controller.update(angular_errors)
        return torch.stack([vs, omegas], dim=1)

    def _compute_target_points(
        self, state_batch: torch.Tensor, reference_paths_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the target points on the reference paths.

        Parameters:
        - state_batch (torch.Tensor): The batch of current states of the robot.
        - reference_paths_batch (torch.Tensor): The batch of paths to follow.

        Returns:
        - target_points (torch.Tensor): The target points on the reference paths.
        """
        deltas = reference_paths_batch - state_batch[:, :2].unsqueeze(1)

        distances = torch.sum(deltas ** 2, dim=2)
        angles = torch.atan2(deltas[:, :, 1], deltas[:, :, 0]) - state_batch[
            :, 2
        ].unsqueeze(1)

        valid_mask = (angles.abs() < torch.pi / 2) & (
            distances > self._lookahead_distance ** 2
        )

        target_indices = valid_mask.long().argmax(dim=1)
        batch_indices = torch.arange(
            state_batch.size(0), device=self._device, dtype=torch.long
        )
        target_indices = torch.where(
            valid_mask.any(dim=1),
            target_indices,
            (reference_paths_batch.size(1) - 1) * torch.ones_like(target_indices),
        )
        target_points = reference_paths_batch[batch_indices, target_indices]
        return target_points

    def _calculate_errors(
        self, state_batch: torch.Tensor, target_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the error signals for the PID controllers.

        Parameters:
        - state_batch (torch.Tensor): The batch of current states of the robot.
        - target_points (torch.Tensor): The target points on the reference paths.

        Returns:
        - linear_errors (torch.Tensor): The error signals for the linear velocity controller.
        - angular_errors (torch.Tensor): The error signals for the angular velocity controller.
        """
        linear_errors = torch.norm(target_points - state_batch[:, :2], dim=1)
        angles_to_targets = torch.atan2(
            target_points[:, 1] - state_batch[:, 1],
            target_points[:, 0] - state_batch[:, 0],
        )
        angular_errors = angles_to_targets - state_batch[:, 2]
        angular_errors = ((angular_errors + torch.pi) % (2 * torch.pi)) - torch.pi
        return linear_errors, angular_errors
