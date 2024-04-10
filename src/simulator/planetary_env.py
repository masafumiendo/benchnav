"""
Kohei Honda and Masafumi Endo, 2024.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import imageio
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from src.environments.grid_map import GridMap
from src.simulator.problem_formulation.robot_model import UnicycleModel
from src.simulator.problem_formulation.utils import ModelConfig
from src.planners.global_planners.sampling_based.tree import Tree
from src.utils.utils import set_randomness


class PlanetaryEnv(gym.Env[torch.Tensor, torch.Tensor]):
    def __init__(
        self,
        grid_map: GridMap,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        delta_t: Optional[float] = 0.1,
        time_limit: Optional[float] = 100,
        stuck_threshold: Optional[float] = 0.1,
        render_mode: Optional[str] = "human",
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
        - time_limit (Optional[float]): Time limit for the episode [s].
        - stuck_threshold (Optional[float]): Threshold for the robot to be considered stuck (low traversability).
        - render_mode (Optional[str]): Rendering mode for the environment.
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

        # Set random seed for reproducibility
        set_randomness(self._seed)

        # Define time step, time limit, and elapsed time
        self._delta_t = delta_t
        self._time_limit = time_limit
        self._elapsed_time = 0

        # Set stuck threshold
        self.stuck_threshold = stuck_threshold

        # Set grid map
        self._grid_map = grid_map

        # Initialize unicycle model with traversability model in observation mode
        self._model_config = ModelConfig(mode="observation")
        self._dynamics = UnicycleModel(
            self._grid_map, self._model_config, self._dtype, self._device
        )

        # Initialize start and goal positions
        self._start_pos = self._initialize_position(start_pos)
        self._goal_pos = self._initialize_position(goal_pos)

        # Initialize robot state and reward
        self._robot_state = self._initialize_robot_state()
        self._reward = torch.full(
            (1,), torch.nan, dtype=self._dtype, device=self._device
        )

        # Set rendering mode and initialize rendering
        if render_mode not in ["human", "rgb_array"]:
            raise ValueError("Invalid rendering mode. Choose 'human' or 'rgb_array'.")
        self._render_mode = render_mode
        self._rendered_frames = []

    def _initialize_position(self, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Initializes a position tensor, either start or goal.
        
        Parameters:
        - pos (Optional[torch.Tensor]): Position tensor.

        Returns:
        - pos (torch.Tensor): Initialized position tensor.
        """
        if pos is None:
            pos = (
                torch.randint(
                    0, self._grid_map.grid_size, (2,), dtype=self._dtype
                ).float()
                * self._grid_map.resolution
            )
        pos = pos.to(self._device)
        # Check if the position is traversable
        if self.collision_check(pos.unsqueeze(0).unsqueeze(0)):
            raise ValueError("Start or goal position is not traversable.")
        return pos

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
        self._reward = torch.full(
            (1,), torch.nan, dtype=self._dtype, device=self._device
        )

        # Initialize history of robot states and rewards with empty numpy arrays
        self._history = {"states": np.empty((0, 3)), "rewards": np.empty((0,))}

        # Reset rendering with two subplots
        self._fig, self._ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        self._ax.set_xlim(self._grid_map.x_limits)
        self._ax.set_ylim(self._grid_map.y_limits)
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        self._ax.set_aspect("equal")

        # Set colormap and normalization for rendering
        self._norm = mcolors.Normalize(vmin=0, vmax=1)
        self._cmap = cm.get_cmap("turbo")

        self._rendered_frames = []

        return self._robot_state

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
        next_state, trav = self._dynamics.transit(
            self._robot_state.unsqueeze(0), action.unsqueeze(0), self._delta_t
        )

        # Update robot state and reward
        self._robot_state = next_state.squeeze(0)
        self._reward = trav

        # Update elapsed time
        self._elapsed_time += self._delta_t

        # Check if episode is terminated or truncated
        is_terminated = torch.norm(self._robot_state[:2] - self._goal_pos) < 0.1  # [m]
        is_truncated = self._elapsed_time > self._time_limit
        return self._robot_state, self._reward, is_terminated, is_truncated

    def collision_check(self, states: torch.Tensor) -> torch.Tensor:
        """
        Check for collisions at the given position.

        Parameters:
        - states (torch.Tensor): States of the robot as batch of position tensors shaped [batch_size, num_positions, 3].

        Returns:
        - is_collisions (torch.Tensor): Collision states at the given position.
        """
        trav = self._dynamics.get_traversability(states)
        return trav <= self.stuck_threshold

    def render(
        self,
        trajectory: torch.Tensor,
        is_collisions: torch.Tensor,
        top_samples: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        reference_paths: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        tree: Optional[Tree] = None,
    ) -> None:
        """
        Render the environment with the trajectory, top samples, and collision states.

        Parameters:
        - trajectory (torch.Tensor): Trajectory of the robot.
        - is_collisions (torch.Tensor): Collision states.
        - top_samples (Optional[tuple[torch.Tensor, torch.Tensor]]): Top samples from the planner.
        - reference_paths (Optional[Union[torch.Tensor, list[torch.Tensor]]]): Reference paths of the robot.
        - tree (Optional[Tree]): Tree data structure from the planner.
        """
        # Plot static information (terrain background maps and robot start and goal positions)
        self._ax.imshow(
            self._grid_map.tensors["colors"].cpu().numpy().transpose(1, 2, 0),
            origin="lower",
            zorder=1,
            extent=self._grid_map.x_limits + self._grid_map.y_limits,
        )
        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            s=25,
            c="red",
            marker="o",
            zorder=2,
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            s=25,
            c="magenta",
            marker="x",
            zorder=2,
        )

        # Plot dynamic trajectory, top samples, and collision states
        self._render_dynamic_information(
            trajectory, is_collisions, top_samples, reference_paths, tree, self._ax
        )

        self._ax.set_xlim(self._grid_map.x_limits)
        self._ax.set_ylim(self._grid_map.y_limits)
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        self._ax.set_aspect("equal")

        # Append rendered frames for animation
        if self._render_mode == "human":
            plt.pause(0.001)
            plt.cla()
        elif self._render_mode == "rgb_array":
            self._fig.canvas.draw()
            frame = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()
            self._rendered_frames.append(frame)

    def _render_dynamic_information(
        self,
        trajectory: torch.Tensor,
        is_collisions: torch.Tensor,
        top_samples: Optional[tuple[torch.Tensor, torch.Tensor]],
        reference_paths: Optional[Union[torch.Tensor, list[torch.Tensor]]],
        tree: Optional[Tree],
        ax: plt.Axes,
    ) -> None:
        """
        Render the dynamic information (trajectory, top samples, and collision states).

        Parameters:
        - trajectory (torch.Tensor): Trajectory of the robot.
        - is_collisions (torch.Tensor): Collision states.
        - top_samples (Optional[tuple[torch.Tensor, torch.Tensor]]): Top samples from the planner.
        - reference_paths (Optional[Union[torch.Tensor, list[torch.Tensor]]]): Reference paths of the robot.
        - tree (Optional[Tree]): Tree data structure from the planner.
        - ax (plt.Axes): Axes to plot the dynamic information.
        """
        # Current robot state with reward
        robot_state = self._robot_state.cpu().numpy()
        reward = self._reward.cpu().numpy()
        ax.scatter(robot_state[0], robot_state[1], c="green", marker="o", zorder=150)

        # Append current robot state and reward to history
        self._history["states"] = np.append(
            self._history["states"], [robot_state], axis=0
        )
        self._history["rewards"] = np.append(self._history["rewards"], reward)

        # History of robot states with rewards
        points = self._history["states"][:, :2]
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        lc = LineCollection(
            segments, cmap=self._cmap, norm=self._norm, alpha=0.8, zorder=100
        )
        lc.set_array(1 - self._history["rewards"])  # slip = 1 - traversability
        ax.add_collection(lc)

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
                zorder=75,
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
                    zorder=50,
                )

        # Reference paths if provided
        if reference_paths is not None:
            if isinstance(reference_paths, torch.Tensor):
                reference_paths = [reference_paths]
            for reference_path in reference_paths:
                ax.plot(
                    reference_path[:, 0].cpu().numpy(),
                    reference_path[:, 1].cpu().numpy(),
                    c="blue",
                    linestyle="--",
                    zorder=25,
                )

        # Tree if provided
        if tree is not None:
            nodes = tree.nodes[: tree.nodes_count].cpu().numpy()
            edges = tree.edges[: tree.nodes_count].cpu().numpy()
            for i in range(1, tree.nodes_count):
                parent = nodes[edges[i]]
                child = nodes[i]
                ax.plot(
                    [parent[0], child[0]],
                    [parent[1], child[1]],
                    c="black",
                    linestyle=":",
                    zorder=15,
                )

                if hasattr(tree, "state_seqs"):
                    if tree.seq_lengths[i] > 0:
                        trajectory = (
                            tree.state_seqs[i, : tree.seq_lengths[i], :2].cpu().numpy()
                        )
                        ax.plot(
                            trajectory[:, 0],
                            trajectory[:, 1],
                            c="lightblue",
                            linestyle="--",
                            zorder=25,
                        )

    def close(self, file_path: str = None) -> None:
        """
        Close the environment and save the animation if file path is provided.

        Parameters:
        - file_path (str): File path to save the animation.
        """
        if len(self._rendered_frames) > 0:
            if file_path is not None:
                file_name = (
                    f"{self._grid_map.instance_name}_{self._seed}.gif"
                    if self._seed
                    else f"{self._grid_map.instance_name}.gif"
                )
                file_path = os.path.join(file_path, file_name)

                directory = os.path.dirname(file_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)

                imageio.mimsave(file_path, self._rendered_frames, fps=10)

        # Close the figure if it exists
        if hasattr(self, "_fig"):
            plt.close(self._fig)
