"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import Module
from pqdict import pqdict
from typing import Optional

from src.environments.grid_map import GridMap


class AStar(Module):
    """
    A class to represent the A* pathfinding algorithm.

    Attributes:
    - heights (np.ndarray): Height map as a 2D array.
    - resolution (float): Resolution of each grid in meters.
    - x_limits (tuple[float, float]): x-axis limits of the grid map.
    - y_limits (tuple[float, float]): y-axis limits of the grid map.
    - travs (np.ndarray): Traversability map as a 2D array.
    - _stuck_threshold (float): Threshold for the robot to be considered stuck (low traversability).
    - device (str): Device to run the algorithm on.
    - _h (int): Height of the grid map.
    - _w (int): Width of the grid map.
    """

    def __init__(
        self,
        grid_map: GridMap,
        travs: torch.Tensor,
        stuck_threshold: float,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the A* pathfinding algorithm.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - travs (torch.Tensor): Predicted traversability map as a 2D tensor.
        - stuck_threshold (float): Threshold for the robot to be considered stuck (low traversability).
        - device (Optional[str]): Device to run the algorithm on.
        """
        super(AStar, self).__init__()
        self.heights = grid_map.tensors["heights"].detach().cpu().numpy()
        self.resolution = grid_map.resolution
        self.x_limits = grid_map.x_limits
        self.y_limits = grid_map.y_limits
        self.travs = travs.detach().cpu().numpy()
        self._stuck_threshold = stuck_threshold
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Check if the traversability and height maps have the same shape
        assert (
            self.travs.shape == self.heights.shape
        ), "Traversability and height maps must have the same shape."
        self._h, self._w = self.heights.shape

    def reset(self, grid_map: GridMap, travs: torch.Tensor) -> None:
        """
        Reset the traversability and height maps for the A* pathfinding algorithm.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - travs (torch.Tensor): Traversability map as a 2D tensor.
        """
        self.heights = grid_map.tensors["heights"].detach().cpu().numpy()
        self.resolution = grid_map.resolution
        self.x_limits = grid_map.x_limits
        self.y_limits = grid_map.y_limits
        self.travs = travs.detach().cpu().numpy()

        # Check if the traversability and height maps have the same shape
        assert (
            self.travs.shape == self.heights.shape
        ), "Traversability and height maps must have the same shape."
        self._h, self._w = self.heights.shape

    def forward(
        self, start_pos: torch.Tensor, goal_pos: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute the A* pathfinding algorithm.

        Parameters:
        - start_pos (torch.Tensor): Start position of the robot as a tensor shaped [2].
        - goal_pos (torch.Tensor): Goal position of the robot as a tensor shaped [2].

        Returns:
        - path (Optional[torch.Tensor]): Path from the start to the goal position as a tensor shaped [num_positions, 2].
        """

        # Convert start and goal positions to grid indices
        start_pos = self._pos_to_index(start_pos)
        goal_pos = self._pos_to_index(goal_pos)

        # Check if the start and goal positions are within bounds
        if not self._is_within_bounds(start_pos) or not self._is_within_bounds(
            goal_pos
        ):
            raise ValueError("Start or goal position is out of bounds.")

        # Check if the start and goal positions are traversable
        if not self._check_collision(start_pos) or not self._check_collision(goal_pos):
            raise ValueError("Start or goal position is not traversable.")

        # Initialize the open and closed sets
        open_set = pqdict({start_pos: 0})
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: self._distance(start_pos, goal_pos)}

        # Search for the path
        while open_set:
            current = open_set.pop()

            if current == goal_pos:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):

                tentative_g_score = g_score[current] + self._distance(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._distance(
                        neighbor, goal_pos
                    )
                    if neighbor not in open_set:
                        open_set[neighbor] = f_score[neighbor]

        return None

    def _distance(self, node1: tuple[int, int], node2: tuple[int, int]) -> float:
        """
        Compute the distance between two positions in 2.5D.

        Parameters:
        - node1 (tuple[int, int]): First position as a grid index.
        - node2 (tuple[int, int]): Second position as a grid index.

        Returns:
        - distance (float): Distance between the two positions.
        """
        dx = abs(node1[0] - node2[0]) * self.resolution
        dy = abs(node1[1] - node2[1]) * self.resolution
        dz = abs(
            self._get_value(node1, self.heights) - self._get_value(node2, self.heights)
        )
        return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    def _get_neighbors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Get the neighbors of a position.

        Parameters:
        - node (tuple[int, int]): Position as a grid index.

        Returns:
        - neighbors (list[tuple[int, int]]): Neighbors of the position.
        """
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        neighbors = []
        for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)
            if self._is_within_bounds(neighbor) and self._check_collision(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def _is_within_bounds(self, node: tuple[int, int]) -> bool:
        """
        Check if a position is within the bounds of the traversability map.

        Parameters:
        - node (tuple[int, int]): Position as a grid index.

        Returns:
        - within_bounds (bool): True if the position is within the bounds, False otherwise.
        """
        # Note that the x and y axes are swapped since the array is in the form of [y, x].
        return 0 <= node[0] < self._w and 0 <= node[1] < self._h

    def _check_collision(self, node: tuple[int, int]) -> bool:
        """
        Check if a position is traversable or not based on the traversability map.

        Parameters:
        - node (tuple[int, int]): Position as a grid index.

        Returns:
        - traversable (bool): True if the position is traversable, False otherwise.
        """
        return self._get_value(node, self.travs) > self._stuck_threshold

    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> torch.Tensor:
        """
        Reconstruct the path from the start to the goal position.

        Parameters:
        - came_from (dict[tuple[int, int], tuple[int, int]]): Dictionary of positions and their parent positions.
        - current (tuple[int, int]): Current position as a grid index.

        Returns:
        - total_path (torch.Tensor): Path from the start to the goal position as a tensor shaped [num_positions, 2].
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return torch.tensor(total_path[::-1], device=self.device) * self.resolution

    def _pos_to_index(self, pos: torch.Tensor) -> tuple[int, int]:
        """
        Convert position to grid index based on the resolution and the bounds of the grid map.

        Parameters:
        - pos (torch.Tensor): Position of the robot as a tensor shaped [2].

        Returns:
        - index (tuple[int, int]): Grid index of the robot.
        """
        return (
            int((pos[0] - self.x_limits[0]) / self.resolution),
            int((pos[1] - self.y_limits[0]) / self.resolution),
        )

    def _get_value(self, node: tuple[int, int], array: np.ndarray) -> float:
        """
        Get the value of a position in an array based on the grid index. 
        Note that the x and y axes are swapped since the array is in the form of [y, x].

        Parameters:
        - node (tuple[int, int]): Position as a grid index.
        - array (np.ndarray): Array to get the value from.

        Returns:
        - value (float): Value of the position in the array.
        """
        return array[node[1], node[0]]
