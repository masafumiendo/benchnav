"""
Kohei Honda and Masafumi Endo, 2024.
"""

from __future__ import annotations
from typing import Optional

import torch
import gymnasium as gym

from src.environments.grid_map import GridMap
from src.robot_models.unicycle_model import UnicycleModel
from src.utils.utils import set_randomness


class PlanetaryEnv(gym.Env[torch.Tensor, torch.Tensor]):
    def __init__(
        self,
        grid_map: GridMap,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        delta_t: Optional[float] = 0.1,
        time_limit: Optional[int] = 100,
        seed: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize PlanetaryEnv class with grid map, starting and goal positions, and other parameters.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensor_data.
        - start_pos (Optional[torch.Tensor]): Starting position of the agent (x, y) [m].
        - goal_pos (Optional[torch.Tensor]): Goal position of the agent (x, y) [m].
        - delta_t (Optional[float]): Time step for simulation [s].
        - time_limit (Optional[int]): Time limit for the episode [s].
        - seed (Optional[int]): Random seed for reproducibility.
        - dtype (Optional[torch.dtype]): Data type for torch tensors.
        - device (Optional[str]): Device to run the model on.
        """
        # Set random seed, data type, and device
        self._seed = seed
        self._dtype = dtype
        self._device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Define time step, time limit, and elapsed time
        self._delta_t = delta_t
        self._time_limit = time_limit
        self._elapsed_time = 0

        # Set grid map and dynamics model
        self._grid_map = grid_map
        self._dynamics = UnicycleModel(self._grid_map, self._dtype, self._device)

        # Set random seed for reproducibility
        set_randomness(self._seed)

        # Initialize start and goal positions and robot state
        self._start_pos = self._initialize_position(start_pos)
        self._goal_pos = self._initialize_position(goal_pos)
        self._robot_state = self._initialize_robot_state()

    def _initialize_position(self, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Initializes a position tensor, either start or goal.
        
        Parameters:
        - pos (Optional[torch.Tensor]): Position tensor.

        Returns:
        - pos (torch.Tensor): Initialized position tensor.
        """
        if pos is not None:
            return pos.to(self._device, self._dtype)
        return (
            torch.randint(
                0,
                self._grid_map.grid_size,
                (2,),
                dtype=self._dtype,
                device=self._device,
            ).float()
            * self._grid_map.resolution
        )

    def _initialize_robot_state(self) -> torch.Tensor:
        """
        Initializes the robot state with position and heading angle.

        Returns:
        - robot_state (torch.Tensor): Robot state tensor (x, y, theta).
        """
        theta = torch.atan2(
            self._goal_pos[1] - self._start_pos[1],
            self._goal_pos[0] - self._start_pos[0],
        )
        return torch.tensor(
            [*self._start_pos, theta], dtype=self._dtype, device=self._device
        )

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, bool, bool]:
        """
        Step the environment.

        Parameters:
        - action (torch.Tensor): Action to take as control input (linear velocity, [m/s] and angular velocity, [rad/s]) 
        as batch of tensors (linear velocity, [m/s] and angular velocity, [rad/s]).

        Returns:
        - robot_state (torch.Tensor): Robot state tensor (x, y, theta) after taking the best action.
        - is_terminated (bool): Whether the episode is terminated (goal reached or not).
        - is_truncated (bool): Whether the episode is truncated (time limit reached or not).
        """
        # Update robot state
        self._robot_state = self._dynamics.transit(
            self._robot_state.unsqueeze(0), action.unsqueeze(0), self._delta_t
        ).squeeze(0)

        # Update elapsed time
        self._elapsed_time += self._delta_t

        # Check if episode is terminated or truncated
        is_terminated = torch.norm(self._robot_state[:2] - self._goal_pos) < 0.1
        is_truncated = self._elapsed_time > self._time_limit
        return self._robot_state, is_terminated, is_truncated

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Reset the environment.
        Parameters:
        - seed (Optional[int]): Random seed for reproducibility.
        - options (Optional[dict]): Additional options.

        Returns:
        - observation (torch.Tensor): Initial observation.
        - info (dict): Additional information.
        """
        raise NotImplementedError

    def render(self) -> None:
        """
        Render the environment.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the environment.
        """
        raise NotImplementedError
