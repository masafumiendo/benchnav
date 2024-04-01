"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch
from torch.nn import Module
from typing import Optional

from src.environments.grid_map import GridMap
from src.utils.utils import set_randomness
from src.planners.global_planners.sampling_based.tree import Tree


class RRT(Module):
    """
    A class to represent the RRT* pathfinding algorithm.

    Attributes:
    - resolution (float): Resolution of each grid in meters.
    - x_limits (tuple[float, float]): x-axis limits of the grid map.
    - y_limits (tuple[float, float]): y-axis limits of the grid map.
    - device (str): Device to run the algorithm on.
    - max_iterations (int): Maximum number of iterations for the algorithm.
    - delta_distance (float): Distance between nodes in the tree.
    - goal_sample_rate (float): Probability of sampling the goal position instead of a random position.
    """

    def __init__(
        self,
        grid_map: GridMap,
        goal_pos: torch.Tensor,
        max_iterations: int = 1000,
        delta_distance: float = 5,
        goal_sample_rate: float = 0.1,
        dim_state: int = 2,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize the RRT pathfinding algorithm.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - goal_pos (torch.Tensor): Goal position of the robot as a tensor shape (2,).
        - max_iterations (int): Maximum number of iterations for the algorithm.
        - delta_distance (float): Distance between nodes in the tree.
        - goal_sample_rate (float): Probability of sampling the goal position instead of a random position.
        - dim_state (int): Dimension of the state space.
        - device (Optional[str]): Device to run the algorithm on.
        - seed (int): Seed for numpy and torch random number generators.
        """
        super(RRT, self).__init__()

        # Set the random seed
        set_randomness(seed)

        self.resolution = grid_map.resolution
        self.x_limits = grid_map.x_limits
        self.y_limits = grid_map.y_limits
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Initialize the RRT* parameters
        self._max_iterations = max_iterations
        self._delta_distance = delta_distance
        self._goal_sample_rate = goal_sample_rate
        self._dim_state = dim_state

        self.tree = None
        self._planner_name = "rrt"

        # Set the goal position after checking its device compatibility
        if goal_pos.device != self.device:
            self._goal_node = goal_pos.to(self.device)

    def forward(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Find the path from the start position to the goal position using the RRT* algorithm.

        Parameters:
        - state (torch.Tensor): State of the robot as a tensor shape (3,).

        Returns:
        - Optional[torch.Tensor]: Path from the start position to the goal position.
        """
        start_node = state[:2] if state.shape[0] == 3 else state

        # Check if the start and goal positions are within the grid map
        if not self._is_within_bounds(start_node) or not self._is_within_bounds(
            self._goal_node
        ):
            raise ValueError("Start or goal position is out of bounds.")

        # Initialize the tree with the start node
        self.tree = Tree(
            dim_state=self._dim_state,
            planner_name=self._planner_name,
            device=self.device,
        )
        self.tree.add_node(start_node)

        for _ in range(self._max_iterations):
            # Sample a position to expand the tree
            sample_pos = self._sample_position()
            nearest_node_index = self.tree.nearest_neighbor(sample_pos)
            new_node, cost, is_feasible = self._steer(
                self.tree.nodes[nearest_node_index], sample_pos
            )

            if not is_feasible:
                continue

            # Add the new node to the tree
            new_node_index = self.tree.add_node(new_node)
            self.tree.add_edge(nearest_node_index, new_node_index)

            # Update the cost of the new node
            new_cost = self.tree.costs[nearest_node_index] + cost
            self.tree.update_cost(new_node_index, new_cost)

        # Reconstruct the path from the goal to the start position
        goal_reached, self._goal_node_indices = self._is_goal_reached()

        if goal_reached:
            return self._reconstruct_path(self._goal_node_indices[0])
        else:
            return None

    def _is_within_bounds(self, node: torch.Tensor) -> bool:
        """
        Check if the given position is within the bounds of the grid map using PyTorch tensors.

        Parameters:
        - node (torch.Tensor): Node position to check, expected shape [2].

        Returns:
        - bool: True if the position is within the bounds, False otherwise.
        """
        x, y = node
        return (
            self.x_limits[0] <= x.item() <= self.x_limits[1]
            and self.y_limits[0] <= y.item() <= self.y_limits[1]
        )

    def _sample_position(self) -> torch.Tensor:
        """
        Sample a random position within the bounds of the grid map.

        Returns:
        - torch.Tensor: Random position within the bounds of the grid map.
        """
        if torch.rand(1) < self._goal_sample_rate:
            return self._goal_node
        else:
            return torch.tensor(
                [
                    torch.rand(1) * (self.x_limits[1] - self.x_limits[0])
                    + self.x_limits[0],
                    torch.rand(1) * (self.y_limits[1] - self.y_limits[0])
                    + self.y_limits[0],
                ],
                device=self.device,
            )

    def _steer(
        self, from_node: torch.Tensor, to_node: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Steer the robot from the nearest node to the sample position.

        Parameters:
        - from_node (torch.Tensor): Nearest node in the tree.
        - to_node (torch.Tensor): Sample position to steer to.

        Returns:
        - torch.Tensor: New node position.
        - torch.Tensor: Distance between the new node and the nearest node.
        - bool: True if the new node is feasible.
        """
        direction = to_node - from_node
        distance = torch.norm(direction)
        if distance > self._delta_distance:
            direction = direction / distance * self._delta_distance
            distance = self._delta_distance
        return from_node + direction, distance, True

    def _is_goal_reached(self, goal_threshold: float = 0.1) -> tuple[bool, list[int]]:
        """
        Check if the goal position is reached from the tree within a threshold.

        Parameters:
        - goal_threshold (float): Threshold for the goal position.

        Returns:
        - tuple[bool, list[int]]: Tuple containing a boolean indicating if the goal is reached, 
                                  and the goal node index.
        """
        distances = torch.norm(
            self.tree.nodes[: self.tree.nodes_count] - self._goal_node, dim=1
        )
        near_goal_indices = torch.where(distances < goal_threshold)[0]

        if near_goal_indices.nelement() == 0:
            return False, []

        goal_node_indices = near_goal_indices[
            torch.argsort(self.tree.costs[near_goal_indices])
        ]
        return True, goal_node_indices.tolist()

    def _reconstruct_path(self, goal_node_index: int) -> torch.Tensor:
        """
        Reconstruct the path from the start to the goal position.

        Parameters:
        - goal_node_index (int): Index of the goal node in the tree.

        Returns:
        - torch.Tensor: Path from the start to the goal position.
        """
        total_path_indices = [goal_node_index]
        current_index = goal_node_index

        while current_index != 0:
            current_index = self.tree.edges[current_index].item()
            total_path_indices.append(current_index)

        total_path_indices.reverse()

        return self.tree.nodes[total_path_indices]
