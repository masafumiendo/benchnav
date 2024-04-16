"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch
from typing import Optional


class Tree:
    def __init__(
        self,
        dim_node: int = 2,
        dim_state: int = 3,
        dim_control: int = 2,
        initial_capacity: int = 1000,
        max_seqs: Optional[int] = 100,
        planner_name: Optional[str] = "rrt",
        device=None,
    ) -> None:
        """
        Initialize the tree data structure.

        Parameters:
        - dim_node (int): Dimension of the node space.
        - dim_state (int): Dimension of the state space.
        - dim_control (int): Dimension of the control space.
        - initial_capacity (int): Initial capacity of the tree.
        - max_seqs (int): Maximum number of sequences to store.
        - planner_name (str): Type of planner to use.
        - device (torch.device): Device to use.

        Attributes:
        - nodes (torch.Tensor): Nodes in the tree as a 2D array.
        - nodes_count (int): Number of nodes in the tree.
        - edges (torch.Tensor): Edges connecting the nodes in the tree.
        - costs (torch.Tensor): Costs of the nodes in the tree.
        - action_seqs (torch.Tensor): Action sequences from the parent to the child node.
        - state_seqs (torch.Tensor): State sequences from the parent to the child node.
        - seq_lengths (torch.Tensor): Lengths of the sequences.
        """
        self._dim_node = dim_node
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._initial_capacity = initial_capacity
        self._max_seqs = max_seqs
        self._device = (
            device if device is None else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Check planner type
        if planner_name not in ["rrt", "cl_rrt"]:
            raise ValueError("Invalid planner name. Choose either 'rrt' or 'rrt_star'.")
        self._planner_name = planner_name

        # Basic tree structure
        self.nodes = torch.zeros(
            (initial_capacity, self._dim_node), device=self._device
        )
        self.nodes_count = 0
        self.edges = -torch.ones(
            initial_capacity, dtype=torch.int64, device=self._device
        )
        self.costs = torch.full((initial_capacity,), torch.inf, device=self._device)
        self.costs[0] = 0

        # Tensors for storing action and state sequences
        if self._planner_name == "cl_rrt":
            self.action_seqs = torch.zeros(
                (initial_capacity, max_seqs, self._dim_control), device=self._device
            )
            self.state_seqs = torch.zeros(
                (initial_capacity, max_seqs + 1, self._dim_state), device=self._device
            )
            self.seq_lengths = torch.zeros(
                initial_capacity, dtype=torch.int64, device=self._device
            )
            self.controllers_states = torch.zeros(
                (initial_capacity, 2 * self._dim_control), device=self._device
            )

    def _ensure_capacity(self) -> None:
        """
        Ensure the capacity of the tree.
        """
        if self.nodes_count >= self.nodes.size(0):
            new_capacity = self.nodes.size(0) * 2  # Double the capacity
            self.nodes = torch.cat(
                (
                    self.nodes,
                    torch.zeros(
                        (new_capacity - self.nodes.size(0), self._dim_node),
                        device=self._device,
                    ),
                )
            )
            self.edges = torch.cat(
                (
                    self.edges,
                    -torch.ones(
                        new_capacity - self.edges.size(0),
                        dtype=torch.int64,
                        device=self._device,
                    ),
                )
            )
            self.costs = torch.cat(
                (
                    self.costs,
                    torch.full(
                        (new_capacity - self.costs.size(0),),
                        torch.inf,
                        device=self._device,
                    ),
                )
            )

            if self._planner_name == "cl_rrt":
                self.action_seqs = torch.cat(
                    (
                        self.action_seqs,
                        torch.zeros(
                            (
                                new_capacity - self.action_seqs.size(0),
                                self._max_seqs,
                                self._dim_control,
                            ),
                            device=self._device,
                        ),
                    )
                )
                self.state_seqs = torch.cat(
                    (
                        self.state_seqs,
                        torch.zeros(
                            (
                                new_capacity - self.state_seqs.size(0),
                                self._max_seqs + 1,
                                self._dim_state,
                            ),
                            device=self._device,
                        ),
                    )
                )
                self.seq_lengths = torch.cat(
                    (
                        self.seq_lengths,
                        torch.zeros(
                            new_capacity - self.seq_lengths.size(0),
                            dtype=torch.int64,
                            device=self._device,
                        ),
                    )
                )
                self.controllers_states = torch.cat(
                    (
                        self.controllers_states,
                        torch.zeros(
                            (
                                new_capacity - self.controllers_states.size(0),
                                2 * self._dim_control,
                            ),
                            device=self._device,
                        ),
                    )
                )

    def add_node(
        self, node: torch.Tensor, controllers_state: Optional[torch.Tensor] = None
    ) -> int:
        """
        Add a node to the tree.

        Parameters:
        - node (torch.Tensor): Node to add.
        - controllers_state (torch.Tensor): State of the given controllers.

        Returns:
        - int: Index of the added node.
        """
        self._ensure_capacity()
        node_index = self.nodes_count
        self.nodes[node_index] = node
        self.nodes_count += 1

        if controllers_state is not None and self._planner_name == "cl_rrt":
            self.controllers_states[node_index] = controllers_state
        return node_index

    def add_edge(
        self,
        parent_index: int,
        child_index: int,
        action_seq: Optional[torch.Tensor] = None,
        state_seq: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add an edge between two nodes in the tree.

        Parameters:
        - parent_index (int): Index of the parent node.
        - child_index (int): Index of the child node.
        - action_seq (torch.Tensor): Action sequence from the parent to the child node.
        - state_seq (torch.Tensor): State sequence from the parent to the child node.
        """
        self.edges[child_index] = parent_index

        if action_seq is not None and state_seq is not None:
            seq_length = action_seq.size(1)
            self.action_seqs[child_index, :seq_length] = action_seq
            self.state_seqs[child_index, : seq_length + 1] = state_seq
            self.seq_lengths[child_index] = seq_length

    def update_cost(self, node_index: int, cost: float) -> None:
        """
        Update the cost of the given node.

        Parameters:
        - node_index (int): Index of the node to update.
        - cost (float): New cost of the node.
        """
        self.costs[node_index] = cost

    def nearest_neighbor(self, new_point: torch.Tensor) -> int:
        """
        Find the index of the nearest neighbor to the given point.

        Parameters:
        - new_point (torch.Tensor): Point to find the nearest neighbor to.

        Returns:
        - int: Index of the nearest neighbor.
        """
        distances = torch.norm(
            self.nodes[: self.nodes_count, :2] - new_point[:2].unsqueeze(0), dim=1
        )
        return torch.argmin(distances).item()
