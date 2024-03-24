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
    def __init__(self, dim: int = 2, initial_capacity: int = 1000) -> None:
        """
        Initialize the tree data structure.

        Parameters:
        - dim (int): Dimension of the tree.
        - initial_capacity (int): Initial capacity of the tree.

        Attributes:
        - nodes (np.ndarray): Nodes in the tree as a 2D array.
        - nodes_count (int): Number of nodes in the tree.
        - edges (np.ndarray): Edges connecting the nodes in the tree.
        - costs (np.ndarray): Costs of the nodes in the tree.
        - kd_tree (KDTree): KDTree for fast nearest neighbor search.
        """
        self._dim = dim
        self.nodes = np.zeros((initial_capacity, dim))
        self.nodes_count = 0
        self.edges = -np.ones(initial_capacity, dtype=int)
        self.costs = np.full(initial_capacity, np.inf)
        self.costs[0] = 0
        self.kd_tree = KDTree(np.empty((0, dim)))

    def _ensure_capacity(self) -> None:
        """
        Ensure the capacity of the tree is sufficient.
        """
        if self.nodes_count + 1 > self.nodes.shape[0]:
            capacity = max(self.nodes_count + 1, int(self.nodes.shape[0] * 1.5))
            self.nodes = np.resize(self.nodes, (capacity, self._dim))
            self.edges = np.resize(self.edges, capacity)
            self.edges[self.nodes_count :] = -1
            self.costs = np.resize(self.costs, capacity)
            self.costs[self.nodes_count :] = np.inf

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
        self._ensure_capacity()
        self.nodes[self.nodes_count, :] = node
        self.nodes_count += 1
        self.kd_tree = KDTree(self.nodes[: self.nodes_count, :])

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

            if self._is_within_bounds(new_node) and not self._is_collisions(
                nearest_node[np.newaxis, :], new_node[np.newaxis, :]
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
        if near_node_indices.size == 0:
            return 0, 0

        near_nodes = self.tree.nodes[near_node_indices]
        new_node = new_node[np.newaxis, :]

        # Check for collisions along the paths from the nearest nodes to the new node
        is_collisions = self._is_collisions(near_nodes, new_node)
        valid_indices = near_node_indices[~is_collisions]
        if valid_indices.size == 0:
            return 0, 0

        # Calculate the total costs for the valid indices
        distances = self._distances(self.tree.nodes[valid_indices], new_node)
        total_costs = self.tree.costs[valid_indices] + distances

        # Choose the parent node with the minimum cost
        min_cost_index = np.argmin(total_costs)
        return valid_indices[min_cost_index], total_costs[min_cost_index]

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
        if near_node_indices.size == 0:
            return

        near_node_indices = near_node_indices[near_node_indices != new_node_index]

        near_nodes = self.tree.nodes[near_node_indices]
        new_node = self.tree.nodes[new_node_index][np.newaxis, :]

        # Check for collisions along the paths from the new node to the nearest nodes
        is_collisions = self._is_collisions(near_nodes, new_node)
        valid_indices = near_node_indices[~is_collisions]

        if valid_indices.size == 0:
            return

        # Calculate the total costs for the valid indices
        distances = self._distances(self.tree.nodes[valid_indices], new_node)
        potential_costs = self.tree.costs[new_node_index] + distances

        is_rewired = potential_costs < self.tree.costs[valid_indices]
        self.tree.edges[valid_indices[is_rewired]] = new_node_index
        self.tree.costs[valid_indices[is_rewired]] = potential_costs[is_rewired]

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

    def _distances(self, nodes1: np.ndarray, nodes2: np.ndarray) -> np.ndarray:
        """
        Calculate the total 3D distances between two sets of nodes.

        Parameters:
        - nodes1 (np.ndarray): Starting nodes.
        - nodes2 (np.ndarray): Ending nodes.

        Returns:
        - np.ndarray: Total 3D distances between the two sets of nodes.
        """
        pose_indices = self._edges_to_pose_indices(nodes1, nodes2)
        valid_mask = pose_indices[:, :, 0] != -1

        x = np.where(valid_mask, pose_indices[:, :, 0], np.nan)
        y = np.where(valid_mask, pose_indices[:, :, 1], np.nan)

        heights = np.where(
            valid_mask,
            self.heights[pose_indices[:, :, 1], pose_indices[:, :, 0]],
            np.nan,
        )
        vertical_distances = np.abs(np.diff(heights, axis=1, prepend=np.nan))

        dx = np.diff(x, axis=1, prepend=np.nan) * self.resolution
        dy = np.diff(y, axis=1, prepend=np.nan) * self.resolution

        return np.nansum(np.sqrt(dx ** 2 + dy ** 2 + vertical_distances ** 2), axis=1)

    def _is_collisions(self, nodes1: np.ndarray, nodes2: np.ndarray) -> np.ndarray:
        """
        Check if the robot is stuck along the paths from the nearest nodes to the new nodes.

        Parameters:
        - nodes1 (np.ndarray): Nearest nodes in the tree.
        - nodes2 (np.ndarray): New nodes to check for collisions.

        Returns:
        - np.ndarray: Boolean array indicating collision presence along each path.
        """
        pose_indices = self._edges_to_pose_indices(nodes1, nodes2)
        valid_mask = pose_indices[:, :, 0] != -1

        travs = np.where(
            valid_mask, self.travs[pose_indices[:, :, 1], pose_indices[:, :, 0]], np.inf
        )
        return np.any(travs <= self._stuck_threshold, axis=1)

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

    def _edges_to_pose_indices(
        self, nodes1: np.ndarray, nodes2: np.ndarray
    ) -> np.ndarray:
        """
        Convert the edges between two sets of nodes to a list of grid indices.

        Parameters:
        - nodes1 (np.ndarray): Starting nodes.
        - nodes2 (np.ndarray): Ending nodes.

        Returns:
        - np.ndarray: Padded grid indices along the edges.
        """

        if nodes1.shape[0] != nodes2.shape[0]:
            if nodes1.shape[0] == 1:
                nodes1 = np.repeat(nodes1, nodes2.shape[0], axis=0)
            elif nodes2.shape[0] == 1:
                nodes2 = np.repeat(nodes2, nodes1.shape[0], axis=0)
            else:
                raise ValueError(
                    "nodes1 and nodes2 have incompatible shapes and neither is a single node."
                )

        x0s, y0s = (
            np.floor(
                (nodes1 - np.array([self.x_limits[0], self.y_limits[0]])[None, :])
                / self.resolution
            )
            .astype(int)
            .T
        )
        x1s, y1s = (
            np.floor(
                (nodes2 - np.array([self.x_limits[0], self.y_limits[0]])[None, :])
                / self.resolution
            )
            .astype(int)
            .T
        )

        # Precompute the maximum number of indices to allocate
        max_len = (
            max(
                [
                    np.abs(x1 - x0) + np.abs(y1 - y0)
                    for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s)
                ]
            )
            + 1
        )

        pose_indices = np.full((len(x0s), max_len, 2), -1, dtype=int)
        for i, (x0, y0, x1, y1) in enumerate(zip(x0s, y0s, x1s, y1s)):
            pose_indice = self._bresenham_line(x0, y0, x1, y1)
            pose_indices[i, : pose_indice.shape[0]] = pose_indice

        return pose_indices

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

    def _is_goal_reached(self, goal_threshold: float = 0.1) -> tuple[bool, list[int]]:
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

        if not near_goal_indices:
            return False, []

        near_goal_nodes = self.tree.nodes[near_goal_indices]
        is_collisions = self._is_collisions(
            near_goal_nodes, self._goal_node[np.newaxis, :]
        )

        valid_indices = np.array(near_goal_indices)[~is_collisions]

        if valid_indices.size == 0:
            return False, []

        goal_node_indices_costs = self.tree.costs[valid_indices]
        goal_node_indices = valid_indices[np.argsort(goal_node_indices_costs)]

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
            current_index = self.tree.edges[current_index]
            total_path_indices.append(current_index)

        total_path_indices.reverse()
        total_path = np.array([self.tree.nodes[i] for i in total_path_indices])

        return torch.from_numpy(total_path).to(self.device, dtype=torch.float32)
