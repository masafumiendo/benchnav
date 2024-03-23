"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
import torch
from torch.nn import Module
from typing import Optional

from src.environments.grid_map import GridMap
from src.simulator.robot_model import UnicycleModel
from src.utils.utils import set_randomness


class Tree:
    def __init__(self, dim: int = 2) -> None:
        """
        Initialize the tree data structure.

        Parameters:
        - dim (int): Dimension of the tree.

        Attributes:
        - nodes (np.ndarray): Nodes in the tree as a 2D array.
        - nodes_count (int): Number of nodes in the tree.
        - edges (dict[int, int]): Edges in the tree.
        - costs (dict[int, float]): Costs of the nodes in the tree.
        - kd_tree (KDTree): KDTree for fast nearest neighbor search.
        """
        self.nodes: np.ndarray = np.zeros((0, dim))
        self.nodes_count: int = 0
        self.edges: dict[int, int] = {}  # {child_index: parent_index}
        self.costs: dict[int, float] = {0: 0}  # {child_index: cost}
        self.kd_tree: KDTree = KDTree(self.nodes)

    def connect_nodes(self, parent_index: int, child_pos: np.ndarray) -> None:
        """
        Connect the parent node to the child node.

        Parameters:
        - parent_index (int): Index of the parent node.
        - child_pos (np.ndarray): Position of the child node.
        """
        self.add_node(child_pos)
        self.add_edge(parent_index, self.nodes_count - 1)

    def add_node(self, node: np.ndarray) -> int:
        """
        Add a node to the tree.

        Parameters:
        - node (np.ndarray): Node to add.
        """
        self.nodes = np.vstack((self.nodes, node))
        self.nodes_count += 1
        self.kd_tree = KDTree(self.nodes)

    def add_edge(self, parent_index: int, child_index) -> None:
        """
        Add an edge to the tree.

        Parameters:
        - parent_index (int): Index of the parent node.
        - child_index (int): Index of the child node.
        """
        self.edges[child_index] = parent_index

    def update_cost(self, node_index, cost: float) -> None:
        """
        Update the cost of the node.

        Parameters:
        - node_index (int): Index of the node.
        - cost (float): Cost to update.
        """
        self.costs[node_index] = cost


class RRTStar(Module):
    """
    A class to represent the RRT* pathfinding algorithm.

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
    - max_iterations (int): Maximum number of iterations for the algorithm.
    - delta_distance (float): Distance between nodes in the tree.
    - goal_sample_rate (float): Probability of sampling the goal position instead of a random position.
    - search_radiues (float): Radius for searching the nearest node in the tree.
    """

    def __init__(
        self,
        grid_map: GridMap,
        goal_pos: torch.Tensor,
        dynamics: UnicycleModel,
        stuck_threshold: float,
        max_iterations: int = 1000,
        delta_distance: float = 2.5,
        goal_sample_rate: float = 0.1,
        search_radius: float = 5,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize the RRT* pathfinding algorithm.

        Parameters:
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - goal_pos (torch.Tensor): Goal position of the robot as a tensor shape (2,).
        - dynamics (UnicycleModel): Unicycle model for the robot.
        - stuck_threshold (float): Threshold for the robot to be considered stuck (low traversability).
        - max_iterations (int): Maximum number of iterations for the algorithm.
        - delta_distance (float): Distance between nodes in the tree.
        - goal_sample_rate (float): Probability of sampling the goal position instead of a random position.
        - search_radius (float): Radius for searching the nearest node in the tree.
        - device (Optional[str]): Device to run the algorithm on.
        - seed (int): Seed for numpy and torch random number generators.
        """
        super(RRTStar, self).__init__()

        # Set the random seed
        set_randomness(seed)

        self.heights = grid_map.tensors["heights"].detach().cpu().numpy()
        self.resolution = grid_map.resolution
        self.x_limits = grid_map.x_limits
        self.y_limits = grid_map.y_limits
        self.travs = dynamics._traversability_model._risks.detach().cpu().numpy()
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
            self.heights.shape == self.travs.shape
        ), "The traversability and height maps must have the same shape."
        self._h, self._w = self.heights.shape

        # Initialize the RRT* parameters
        self._max_iterations = max_iterations
        self._delta_distance = delta_distance
        self._goal_sample_rate = goal_sample_rate
        self._search_radius = search_radius

        self.tree = None

        # Set the goal position
        self._goal_node = goal_pos.detach().cpu().numpy()
        self._goal_node_indices = []

    def forward(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Find the path from the start position to the goal position using the RRT* algorithm.

        Parameters:
        - state (torch.Tensor): State of the robot as a tensor shape (3,).

        Returns:
        - Optional[torch.Tensor]: Path from the start position to the goal position.
        """
        state = state[:2] if state.shape[0] == 3 else state
        start_node = state.detach().cpu().numpy()

        # Check if the start and goal positions are within the grid map
        if not self._is_within_bounds(start_node) or not self._is_within_bounds(
            self._goal_node
        ):
            raise ValueError("Start or goal position is out of bounds.")

        # Initialize the tree with the start node
        self.tree = Tree(dim=2)
        self.tree.add_node(start_node)

        for _ in range(self._max_iterations):
            # Sample a position to expand the tree
            sample_pos = self._sample_position()
            nearest_node = self._nearest_node(sample_pos)
            new_node = self._steer(nearest_node, sample_pos)

            if self._is_within_bounds(new_node) and not self._is_collision(
                nearest_node, new_node
            ):
                # Connect the new node to the tree
                near_node_indices = self._near_node_indices(new_node)
                parent_index, cost = self._choose_parent(near_node_indices, new_node)
                self.tree.connect_nodes(parent_index, new_node)

                # Update the cost of the new node and rewire the tree
                new_node_index = self.tree.nodes_count - 1
                self.tree.update_cost(new_node_index, cost)
                self._rewire(near_node_indices, new_node_index)

        # Reconstruct the path from the goal to the start position
        goal_reached, self._goal_node_indices = self._is_goal_reached()
        if goal_reached:
            return self._reconstruct_path(self._goal_node_indices[0])
        else:
            return None

    def _sample_position(self) -> np.ndarray:
        """
        Sample a random position within the bounds of the grid map.

        Returns:
        - np.ndarray: Random position or the goal position.
        """
        if np.random.rand() < self._goal_sample_rate:
            return self._goal_node
        else:
            return np.random.uniform(
                [self.x_limits[0], self.y_limits[0]],
                [self.x_limits[1], self.y_limits[1]],
            )

    def _nearest_node(self, node: np.ndarray) -> np.ndarray:
        """
        Find the nearest node in the tree to the given position.

        Parameters:
        - node (np.ndarray): Position to find the nearest node to.

        Returns:
        - np.ndarray: Nearest node in the tree.
        """
        return self.tree.nodes[self.tree.kd_tree.query(node)[1]]

    def _near_node_indices(self, node: np.ndarray) -> np.ndarray:
        """
        Find the nodes within the search radius to the given position.

        Parameters:
        - node (np.ndarray): Position to find the nodes within the search radius to.

        Returns:
        - np.ndarray: Indices of the nodes within the search radius.
        """
        if self.tree.nodes_count == 1:
            return np.array([0])

        search_radius = self._search_radius * np.sqrt(
            np.log(self.tree.nodes_count) / self.tree.nodes_count
        )
        near_node_indices = self.tree.kd_tree.query_ball_point(node, search_radius)
        return np.array(near_node_indices)

    def _choose_parent(
        self, near_node_indices: np.ndarray, new_node: np.ndarray
    ) -> tuple[int, float]:
        """
        Choose the parent node for the new node based on the cost of the nodes within the search radius.

        Parameters:
        - near_node_indices (np.ndarray): Indices of the nodes within the search radius.
        - new_node (np.ndarray): New node to choose the parent for.

        Returns:
        - tuple[int, float]: Index of the parent node and the cost of the new node.
        """
        if len(near_node_indices) == 0:
            return 0, 0

        valid_indices = [
            i
            for i in near_node_indices
            if not self._is_collision(self.tree.nodes[i], new_node)
        ]

        if len(valid_indices) == 0:
            return 0, 0

        costs = np.array(
            [
                self.tree.costs[i] + self._distance(self.tree.nodes[i], new_node)
                for i in valid_indices
            ]
        )

        # Choose the parent node with the minimum cost
        min_cost_index = np.argmin(costs)
        return valid_indices[min_cost_index], costs[min_cost_index]

    def _steer(self, nearest_node: np.ndarray, sample_pos: np.ndarray) -> np.ndarray:
        """
        Steer the robot from the nearest node to the sample position.

        Parameters:
        - nearest_node (np.ndarray): Nearest node in the tree.
        - sample_pos (np.ndarray): Sample position to steer to.

        Returns:
        - np.ndarray: New node position.
        """
        direction = sample_pos - nearest_node
        distance = np.linalg.norm(direction)
        if distance > self._delta_distance:
            direction = direction / distance * self._delta_distance
        return nearest_node + direction

    def _rewire(self, near_node_indices: np.ndarray, new_node_index: int) -> None:
        """
        Rewire the tree based on the new node.

        Parameters:
        - near_node_indices (np.ndarray): Indices of the nodes within the search radius.
        - new_node_index (int): Index of the new node.
        """
        new_node = self.tree.nodes[new_node_index]
        new_cost = self.tree.costs[new_node_index]

        for near_index in near_node_indices:
            if near_index == new_node_index:
                continue  # Skip if it's the same node
            near_node = self.tree.nodes[near_index]
            if self._is_collision(near_node, new_node):
                continue  # Skip nodes that would result in a collision

            # Calculate potential new cost for near_node if it were to be connected through new_node
            potential_new_cost = new_cost + self._distance(new_node, near_node)
            if potential_new_cost < self.tree.costs[near_index]:
                # Update parent to new_node if it provides a shorter path
                self.tree.edges[near_index] = new_node_index
                self.tree.costs[near_index] = potential_new_cost

    def _is_within_bounds(self, node: np.ndarray) -> bool:
        """
        Check if the given position is within the bounds of the grid map.

        Parameters:
        - node (np.ndarray): Node position to check.

        Returns:
        - bool: True if the position is within the bounds, False otherwise.
        """
        x, y = node
        return (
            self.x_limits[0] <= x <= self.x_limits[1]
            and self.y_limits[0] <= y <= self.y_limits[1]
        )

    def _distance(self, node1: np.ndarray, node2: np.ndarray) -> float:
        """
        Calculate the total 3D distance between two nodes.

        Parameters:
        - node1 (np.ndarray): Starting node.
        - node2 (np.ndarray): Ending node.

        Returns:
        - float: Total 3D distance between the two nodes.
        """
        pose_indices = self._edge_to_pose_indices(node1, node2)
        heights = self.heights[pose_indices[:, 1], pose_indices[:, 0]]
        vertical_distances = np.abs(np.diff(heights))

        dx = np.diff(pose_indices[:, 0]) * self.resolution
        dy = np.diff(pose_indices[:, 1]) * self.resolution

        return np.sum(np.sqrt(dx ** 2 + dy ** 2 + vertical_distances ** 2))

    def _is_collision(self, node1: np.ndarray, node2: np.ndarray) -> bool:
        """
        Check if the robot is stuck along the path from the nearest node to the new node.

        Parameters:
        - node1 (np.ndarray): Nearest node in the tree.
        - node2 (np.ndarray): New node to check for collisions.

        Returns:
        - bool: True if the robot is stuck along the path, False otherwise.
        """
        pose_indices = self._edge_to_pose_indices(node1, node2)
        return np.any(
            self.travs[pose_indices[:, 1], pose_indices[:, 0]] <= self._stuck_threshold
        )

    def _edge_to_pose_indices(self, node1: np.ndarray, node2: np.ndarray) -> np.ndarray:
        """
        Convert the edge between two nodes to a list of grid indices.

        Parameters:
        - node1 (np.ndarray): Starting node.
        - node2 (np.ndarray): Ending node.

        Returns:
        - np.ndarray: List of grid indices along the edge.
        """
        x0, y0 = np.floor(
            (node1 - np.array([self.x_limits[0], self.y_limits[0]])) / self.resolution
        ).astype(int)
        x1, y1 = np.floor(
            (node2 - np.array([self.x_limits[0], self.y_limits[0]])) / self.resolution
        ).astype(int)

        return self._bresenham_line(x0, y0, x1, y1)

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """
        Generate a list of grid indices along the line between two points.

        Parameters:
        - x0 (int): x-coordinate of the starting point.
        - y0 (int): y-coordinate of the starting point.
        - x1 (int): x-coordinate of the ending point.
        - y1 (int): y-coordinate of the ending point.

        Returns:
        - np.ndarray: List of grid indices along the line.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        indices = []
        while True:
            indices.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return np.array(indices)

    def _is_goal_reached(
        self, goal_threshold: float = 0.1
    ) -> tuple[bool, list[int]]:
        """
        Check if the goal position is reached from the tree within a threshold.

        Parameters:
        - goal_threshold (float): Threshold for the goal position.

        Returns:
        - tuple[bool, list[int]]: Tuple containing a boolean indicating if the goal is reached, 
                                            and the goal node index.
        """
        near_goal_indices = self.tree.kd_tree.query_ball_point(
            self._goal_node, goal_threshold
        )

        goal_node_indices_costs = []
        for index in near_goal_indices:
            if not self._is_collision(self.tree.nodes[index], self._goal_node):
                goal_node_indices_costs.append((index, self.tree.costs[index]))

        if len(goal_node_indices_costs) == 0:
            return False, []
        
        goal_node_indices_costs.sort(key=lambda x: x[1])
        goal_node_indices = [node[0] for node in goal_node_indices_costs]
        return True, goal_node_indices

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
            current_index = self.tree.edges[current_index]
            total_path_indices.append(current_index)

        total_path_indices.reverse()
        total_path = np.array([self.tree.nodes[i] for i in total_path_indices])

        return torch.from_numpy(total_path).to(self.device, dtype=torch.float32)
