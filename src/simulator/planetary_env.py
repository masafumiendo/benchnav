"""
Kohei Honda and Masafumi Endo, 2024.
"""

import os

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from src.environments.grid_map import GridMap
from src.simulator.robot_model import UnicycleModel
from src.simulator.utils import ModelConfig
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
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
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

        # Set grid map
        self._grid_map = grid_map

        # Initialize unicycle model with traversability model in observation mode
        self._model_config = ModelConfig(mode="observation")
        self._dynamics = UnicycleModel(
            self._grid_map, self._model_config, self._dtype, self._device
        )

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

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Reset the environment.
        Parameters:
        - seed (Optional[int]): Random seed for reproducibility.

        Returns:
        - robot_state (torch.Tensor): Robot state tensor (x, y, theta) after reset.
        """
        # Set random seed for reproducibility
        set_randomness(seed)

        # Reset elapsed time
        self._elapsed_time = 0

        # Initialize start and goal positions and robot state
        self._start_pos = self._initialize_position(self._start_pos)
        self._goal_pos = self._initialize_position(self._goal_pos)
        self._robot_state = self._initialize_robot_state()

        # Reset rendering with two subplots
        self._fig, self._ax = plt.subplots(1, 2, tight_layout=True)

        self._ax[0].set_xlim(self._grid_map.x_limits)
        self._ax[0].set_ylim(self._grid_map.y_limits)
        self._ax[0].set_aspect("equal")
        self._ax[0].set_title("Terrain Appearance Map")

        self._ax[1].set_xlim(self._grid_map.x_limits)
        self._ax[1].set_ylim(self._grid_map.y_limits)
        self._ax[1].set_aspect("equal")
        self._ax[1].set_title("Terrain Traversability Map")

        self._rendered_frames = []

        return self._robot_state

    def render(
        self,
        trajectory: torch.Tensor,
        top_samples: tuple[torch.Tensor, torch.Tensor],
        is_collisions: torch.Tensor,
    ) -> None:
        """
        Render the environment with the trajectory, top samples, and collision states.

        Parameters:
        - trajectory (torch.Tensor): Trajectory of the robot.
        - top_samples (tuple[torch.Tensor, torch.Tensor]): Top samples from the planner.
        - is_collisions (torch.Tensor): Collision states.
        """

        # Plot static information (terrain background maps and robot start and goal positions)
        self._ax[0].imshow(
            self._grid_map.tensors["colors"].cpu().numpy().transpose(1, 2, 0),
            extent=self._grid_map.extent,
            origin="lower",
            zorder=10,
        )
        self._ax[1].imshow(
            self._grid_map.tensors["latent_models"].mean.cpu().numpy(),
            extent=self._grid_map.extent,
            origin="lower",
            zorder=10,
        )
        self._ax[0].scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            c="red",
            marker="o",
            zorder=10,
        )
        self._ax[0].scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            c="orange",
            marker="x",
            zorder=10,
        )

        # Plot dynamic trajectory, top samples, and collision states
        self._render_dynamic_information(
            trajectory, top_samples, is_collisions, self._ax[0]
        )
        self._render_dynamic_information(
            trajectory, top_samples, is_collisions, self._ax[1]
        )

    def _render_dynamic_information(
        self,
        trajectory: torch.Tensor,
        top_samples: tuple[torch.Tensor, torch.Tensor],
        is_collisions: torch.Tensor,
        ax: plt.Axes,
    ) -> None:
        """
        Render the dynamic information (trajectory, top samples, and collision states).

        Parameters:
        - trajectory (torch.Tensor): Trajectory of the robot.
        - top_samples (tuple[torch.Tensor, torch.Tensor]): Top samples from the planner.
        - is_collisions (torch.Tensor): Collision states.
        - ax (plt.Axes): Axes to plot the dynamic information.
        """
        # Current robot state
        ax.scatter(
            self._robot_state[0].item(),
            self._robot_state[1].item(),
            c="green",
            marker="o",
            zorder=100,
        )

        # Trajectory if provided
        if trajectory is not None:
            colors = np.array(["darkblue"] * trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = np.any(is_collisions.cpu().numpy(), axis=0)
                colors[is_collisions] = "red"

            ax.scatter(
                trajectory[0, :, 0].cpu().numpy(),
                trajectory[0, :, 1].cpu().numpy(),
                c=colors,
                marker="o",
                s=3,
                zorder=2,
            )

        # Top samples if provided
        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples, top_weights = (
                top_samples.cpu().numpy(),
                top_weights.cpu().numpy(),
            )
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            for i in range(top_samples.shape[0]):
                ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    c="lightblue",
                    alpha=top_weights[i],
                    zorder=1,
                )

    def close(self, file_path: str = None) -> None:
        """
        Close the environment and save the animation if file path is provided.

        Parameters:
        - file_path (str): File path to save the animation.
        """
        if file_path is not None:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
