"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch
from typing import Optional

from src.environments.grid_map import GridMap
from src.utils.utils import set_randomness
from src.simulator.robot_model import UnicycleModel
from src.planners.local_planners.objectives import Objectives
from src.planners.global_planners.sampling_based.tree import Tree
from src.planners.global_planners.sampling_based.rrt import RRT
from src.followers.pure_pursuit import PurePursuit
from src.followers.pure_pursuit import PIDController


class CLRRT(RRT):
    """
    A class to represent closed-loop RRT for global motion planning.
    """

    def __init__(
        self,
        dim_state: int,
        dim_control: int,
        dynamics: UnicycleModel,
        objectives: Objectives,
        grid_map: GridMap,
        delta_t: float,
        max_iterations: int = 1000,
        delta_distance: float = 5,
        goal_sample_rate: float = 0.25,
        max_seqs: int = 100,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        """
        Initialize the RRT pathfinding algorithm.

        Parameters:
        - dim_state (int): Dimension of the state space.
        - dim_control (int): Dimension of the control space.
        - dynamics (UnicycleModel): Unicycle model object for the robot.
        - objectives (Objectives): Objectives object containing the objectives for the robot.
        - grid_map (GridMap): Grid map object containing terrain information as tensors.
        - goal_pos (torch.Tensor): Goal position of the robot as a tensor shape (2,).
        - max_iterations (int): Maximum number of iterations for the algorithm.
        - delta_distance (float): Distance between nodes in the tree.
        - goal_sample_rate (float): Probability of sampling the goal position instead of a random position.
        - max_seqs (int): Maximum number of action sequences to consider.
        - device (Optional[str]): Device to run the algorithm on.
        - dtype (torch.dtype): Data type to run the algorithm on.
        - seed (int): Seed for numpy and torch random number generators.
        """
        super(CLRRT, self).__init__(
            grid_map=grid_map,
            goal_pos=objectives._goal_pos,
            max_iterations=max_iterations,
            delta_distance=delta_distance,
            goal_sample_rate=goal_sample_rate,
            dim_state=dim_state,
            device=device,
            seed=seed,
        )

        set_randomness(seed)

        # Check the dimensions
        assert dynamics.min_action.shape == (
            dim_control,
        ), "minimum actions must be a tensor of shape (dim_control,)"
        assert dynamics.max_action.shape == (
            dim_control,
        ), "maximum actions must be a tensor of shape (dim_control,)"

        # device and dtype
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # Set parameters
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
        self._max_seqs = max_seqs

        self._planner_name = "cl_rrt"

        # Initialize Pure Pursuit controller
        lin_controller = PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            delta_t=delta_t,
            device=self._device,
            dtype=self._dtype,
        )
        ang_controller = PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            delta_t=delta_t,
            device=self._device,
            dtype=self._dtype,
        )
        self._pure_pursuit = PurePursuit(
            lookahead_distance=0.5,
            lin_controller=lin_controller,
            ang_controller=ang_controller,
            device=self._device,
            dtype=self._dtype,
        )

        # Set initial starting state
        self._start_state = None

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the closed-loop RRT algorithm to compute action and state sequences.

        Parameters:
        - state (torch.Tensor): The current state of the robot.

        Returns:
        - optimal_action_seq (torch.Tensor): Optimal action sequence.
        - optimal_state_seq (torch.Tensor): Optimal state sequence.        
        """
        self._start_state = state
        start_node = state[:2] if state.shape[0] == 3 else state

        if not self._is_within_bounds(start_node) or not self._is_within_bounds(
            self._goal_node
        ):
            raise ValueError("Start or goal position is out of bounds.")

        self.tree = Tree(
            dim_state=self._dim_state,
            dim_control=self._dim_control,
            planner_name=self._planner_name,
            device=self._device,
        )
        self.tree.add_node(start_node)

        for _ in range(self._max_iterations):
            sample_pos = self._sample_position()
            nearest_node_index = self.tree.nearest_neighbor(sample_pos)
            new_node, action_seq, state_seq, controllers_state, cost, is_feasible = self._steer(
                nearest_node_index, sample_pos
            )

            if not is_feasible:
                continue

            # Add the new node to the tree
            new_node_index = self.tree.add_node(new_node, controllers_state)
            self.tree.add_edge(
                nearest_node_index, new_node_index, action_seq, state_seq
            )

            new_cost = self.tree.costs[nearest_node_index] + cost
            self.tree.update_cost(new_node_index, new_cost)

        goal_reached, self._goal_node_indices = self._is_goal_reached()

        if goal_reached:
            return self._reconstruct_state_action_seq(self._goal_node_indices[0])
        else:
            return None, None

    def _steer(
        self, from_node_index: int, to_node: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Steer the robot from the nearest node to the sample position.

        Parameters:
        - from_node_index (int): Index of the nearest node in the tree.
        - to_node (torch.Tensor): Sample position to steer to.

        Returns:
        - torch.Tensor: New node in the tree.
        - torch.Tensor: Action sequence.
        - torch.Tensor: State sequence.
        - torch.Tensor: Cost of the path following.
        - bool: Feasibility of the path following.
        """
        # Retrieve the nearest node
        from_node = self.tree.nodes[from_node_index]
        # Restrict the distance between nodes
        direction = to_node - from_node
        distance = torch.norm(direction)
        if distance > self._delta_distance:
            to_node = from_node + self._delta_distance * direction / distance

        # Generate the dense reference path
        reference_path = self._generate_reference_path(from_node, to_node)

        # Simulate the path following
        action_seq, state_seq, controllers_state, cost, is_feasible = self._simulate_path_following(
            from_node_index, reference_path
        )

        if is_feasible:
            return (
                state_seq[:, -1, :2],
                action_seq,
                state_seq,
                controllers_state,
                cost,
                True,
            )
        else:
            return None, None, None, None, None, False

    def _generate_reference_path(
        self, from_node: torch.Tensor, to_node: torch.Tensor, resolution: float = 0.5
    ) -> torch.Tensor:
        """
        Generate the reference path from the nearest node to the sample position.

        Parameters:
        - from_node (torch.Tensor): Nearest node in the tree.
        - to_node (torch.Tensor): Sample position to steer to.
        - resolution (float): Resolution of the reference path.

        Returns:
        - torch.Tensor: Reference path.
        """
        distance = torch.norm(to_node - from_node)
        steps = (distance / resolution).ceil().int().item()

        reference_path_x = torch.linspace(
            from_node[0], to_node[0], steps + 1, device=self._device
        )
        reference_path_y = torch.linspace(
            from_node[1], to_node[1], steps + 1, device=self._device
        )

        return torch.stack([reference_path_x, reference_path_y], dim=1)

    def _simulate_path_following(
        self, start_node_index: int, reference_path: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Simulate the path following using the Pure Pursuit controller.

        Parameters:
        - start_node_index (int): Index of the start node in the tree.
        - reference_path (torch.Tensor): Reference path to follow.

        Returns:
        - action_seq (torch.Tensor): Action sequence.
        - state_seq (torch.Tensor): State sequence.
        - controllers_state (torch.Tensor): Terminal state of the PID controllers.
        - cost (torch.Tensor): Cost of the path following.
        - is_feasible (bool): Feasibility of the path following.
        """
        # Initialize action and state sequences
        action_seq = torch.zeros(
            (1, self._max_seqs, self._dim_control),
            device=self._device,
            dtype=self._dtype,
        )
        state_seq = torch.zeros(
            (1, self._max_seqs + 1, self._dim_state),
            device=self._device,
            dtype=self._dtype,
        )
        cost = torch.tensor([0.0], device=self._device, dtype=self._dtype)

        # Retrieve the initial states from the tree
        if start_node_index == 0:
            state_seq[:, 0, :] = self._start_state.repeat(1, 1)
            self._pure_pursuit.reset(1)
        else:
            seq_length = self.tree.seq_lengths[start_node_index]
            controllers_state = self.tree.controllers_states[start_node_index]

            state_seq[:, 0, :] = self.tree.state_seqs[
                start_node_index, seq_length : seq_length + 1, :
            ]
            self._pure_pursuit.reset(1, controllers_state.unsqueeze(0))

        reference_path = reference_path.unsqueeze(0)
        is_feasible = False

        for t in range(self._max_seqs):
            action_seq[:, t, :], controllers_state = self._pure_pursuit.control(
                state_seq[:, t, :], reference_path
            )
            state_seq[:, t + 1, :] = self._dynamics.transit(
                state_seq[:, t, :], action_seq[:, t, :]
            )

            cost += self._stage_cost(state_seq[:, t, :], action_seq[:, t, :])

            if torch.norm(reference_path[:, -1, :] - state_seq[:, t + 1, :2]) < 1:
                action_seq = action_seq[:, : t + 1, :]
                state_seq = state_seq[:, : t + 2, :]
                cost += self._terminal_cost(state_seq[:, t + 1, :])
                is_feasible = True
                break

        if not is_feasible:
            cost += self._terminal_cost(state_seq[:, -1, :])

        return action_seq, state_seq, controllers_state, cost, is_feasible

    def _reconstruct_state_action_seq(
        self, goal_node_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct the optimal state and action sequences from the goal node.

        Parameters:
        - goal_node_index (int): Index of the goal node in the tree.

        Returns:
        - torch.Tensor: Optimal action sequence.
        - torch.Tensor: Optimal state sequence.
        """
        action_seqs = []
        state_seqs = []

        current_index = goal_node_index
        while current_index != 0:  # Start node
            parent_index = self.tree.edges[current_index]
            seq_length = self.tree.seq_lengths[current_index]

            action_seq = self.tree.action_seqs[current_index, :seq_length]
            state_seq = self.tree.state_seqs[current_index, : seq_length + 1]

            action_seqs.insert(0, action_seq)
            if state_seqs:
                state_seqs.insert(0, state_seq[1:])
            else:
                state_seqs.insert(0, state_seq)

            current_index = parent_index

        optimal_action_seq = torch.cat(action_seqs, dim=0)
        optimal_state_seq = torch.cat(state_seqs, dim=0)

        return optimal_action_seq, optimal_state_seq.unsqueeze(0)
