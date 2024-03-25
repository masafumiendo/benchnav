"""
Masafumi Endo, 2024.
"""

from __future__ import annotations

import torch

from src.planners.global_planners.rrtstar import RRTStar
from src.planners.local_planners.objectives import Objectives
from src.followers.pure_pursuit import PIDController
from src.followers.pure_pursuit import PurePursuit
from src.simulator.robot_model import UnicycleModel


class CLRRTStar(RRTStar):
    """
    Closed Loop RRT* algorithm for global path planning.
    The algorithm first run RRT* and then optimize the path by using a controller to follow the path.    
    """

    def __init__(
        self,
        dim_state: int,
        dim_control: int,
        dynamics: UnicycleModel,
        objectives: Objectives,
        delta_t: float = 0.1,
        time_limit: float = 100,
        max_iterations: int = 1000,
        delta_distance: float = 2.5,
        goal_sample_rate: float = 0.1,
        search_radius: float = 5,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ):
        """
        Initialize the Closed Loop RRT* algorithm.

        Parameters:
        - dim_state (int): The dimension of the state space.
        - dim_control (int): The dimension of the control space.
        - dynamics (UnicycleModel): The dynamics model of the robot.
        - objectives (Objectives): The objectives class containing the stage and terminal costs.
        - delta_t (float): The time step for simulation [s].
        - time_limit (float): The time limit for the CLRRT* simulation [s].
        - max_iterations (int): The maximum number of iterations for the RRT* algorithm.
        - delta_distance (float): The distance between two nodes in the RRT* tree.
        - goal_sample_rate (float): The rate at which the goal is sampled.
        - search_radius (float): The radius within which the nearest node is searched.
        - device (torch.device): The device on which the algorithm is run.
        - dtype (torch.dtype): The data type of the tensors.
        - seed (int): The random seed.
        """
        # Initialize RRT* algorithm
        super(CLRRTStar, self).__init__(
            dynamics._grid_map,
            objectives._goal_pos,
            dynamics,
            objectives._stuck_threshold,
            max_iterations,
            delta_distance,
            goal_sample_rate,
            search_radius,
            device,
            seed,
        )

        # torch seed
        torch.manual_seed(seed)

        # check dimensions
        assert dynamics.min_action.shape == (
            dim_control,
        ), "minimum actions must be a tensor of shape (dim_control,)"
        assert dynamics.max_action.shape == (
            dim_control,
        ), "maximum actions must be a tensor of shape (dim_control,)"

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # set parameters
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._stage_cost = objectives.stage_cost
        self._terminal_cost = objectives.terminal_cost
        self._goal_pos = objectives._goal_pos
        self._u_min = (
            self._dynamics.min_action.clone().detach().to(self._device, self._dtype)
        )
        self._u_max = (
            self._dynamics.max_action.clone().detach().to(self._device, self._dtype)
        )
        self._max_iterations = int(time_limit / delta_t)

        # Initialize Pure Pursuit controller
        lin_controller = PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.2,
            delta_t=delta_t,
            device=self._device,
            dtype=self._dtype,
        )
        ang_controller = PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.2,
            delta_t=delta_t,
            device=self._device,
            dtype=self._dtype,
        )
        self._pure_pursuit = PurePursuit(
            lookahead_distance=5.0,
            lin_controller=lin_controller,
            ang_controller=ang_controller,
            device=self._device,
            dtype=self._dtype,
        )

        # Initialize variables for top samples
        self._state_seq_batch = None
        self._weights = None

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the Closed Loop RRT* algorithm to compute action and state sequences.

        Parameters:
        - state (torch.Tensor): The current state of the robot as a tensor of shape (3,).

        Returns:
        - optimal_action_seq (torch.Tensor): Optimal action sequence.
        - optimal_state_seq (torch.Tensor): Optimal state sequence.
        """
        # Run RRT* algorithm to generate reference paths
        super(CLRRTStar, self).forward(state)
        reference_paths_batch = self._generate_reference_paths()

        optimal_action_seq, optimal_state_seq = self._simulate_path_followings(
            state, reference_paths_batch
        )

        return optimal_action_seq, optimal_state_seq.unsqueeze(0)

    def _generate_reference_paths(self) -> list[torch.Tensor]:
        """
        Generate reference paths from the RRT* tree.

        Returns:
        - reference_paths (list[torch.Tensor]): List of reference paths.
        """
        reference_paths = [
            self._reconstruct_path(goal_node_index)
            for goal_node_index in self._goal_node_indices
        ]

        max_length = max(path.shape[0] for path in reference_paths)

        reference_paths_batch = torch.zeros(
            (len(reference_paths), max_length, 2),
            device=self._device,
            dtype=self._dtype,
        )

        for i, path in enumerate(reference_paths):
            path_length = path.shape[0]
            reference_paths_batch[i, :path_length, :] = path

            if path_length < max_length:
                reference_paths_batch[i, path_length:, :] = path[-1]

        return reference_paths_batch

    def _simulate_path_followings(
        self, initial_state: torch.Tensor, reference_paths_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Simulate path following using a path tracking controller.

        Parameters:
        - initial_state (torch.Tensor): The initial state of the robot.
        - reference_paths_batch (torch.Tensor): The batched reference paths to follow.

        Returns:
        - action_seq_batch (torch.Tensor): The optimal action sequence.
        - state_seq_batch (torch.Tensor): The optimal state sequence.
        - cost (float): The cost of the path.
        """
        num_paths = reference_paths_batch.shape[0]
        state_batch = initial_state.unsqueeze(0).repeat(num_paths, 1)

        # Initialize the batched action and state sequences
        action_seq_batch = torch.zeros(
            (num_paths, self._max_iterations, self._dim_control),
            device=self._device,
            dtype=self._dtype,
        )
        state_seq_batch = torch.zeros(
            (num_paths, self._max_iterations, self._dim_state),
            device=self._device,
            dtype=self._dtype,
        )
        cost_batch = torch.zeros(num_paths, device=self._device, dtype=self._dtype)
        t_ends = torch.full((num_paths,), self._max_iterations, device=self._device)

        # Reset pure pursuit controller for variable batch size
        self._pure_pursuit.reset(num_paths)

        for t in range(self._max_iterations):
            action_batch = self._pure_pursuit.control(
                state_batch, reference_paths_batch
            )
            next_state_batch = self._dynamics.transit(state_batch, action_batch)

            cost_batch += self._stage_cost(state_batch, action_batch)

            state_seq_batch[:, t, :] = state_batch
            action_seq_batch[:, t, :] = action_batch

            state_batch = next_state_batch

            goal_reached = torch.norm(state_batch[:, :2] - self._goal_pos, dim=1) < 0.1
            t_ends = torch.where(
                goal_reached & (t_ends == self._max_iterations),
                torch.tensor(t, device=self._device),
                t_ends,
            )

            if goal_reached.all():
                break

        cost_batch += self._terminal_cost(state_batch)

        min_cost_index = torch.argmin(cost_batch)
        t_end = t_ends[min_cost_index]

        # Extract the optimal action and state sequences
        optimal_action_seq = action_seq_batch[min_cost_index, : t_end + 1, :]
        optimal_state_seq = state_seq_batch[min_cost_index, : t_end + 1, :]

        # Store the state sequence for top samples
        self._state_seq_batch = state_seq_batch
        self._weights = torch.softmax(-cost_batch, dim=0)
        return optimal_action_seq, optimal_state_seq

    def get_top_samples(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top samples from the simulation.

        Returns:
        - top_samples (torch.Tensor): The top samples.
        - top_weights (torch.Tensor): The top weights.
        """
        top_indices = torch.argsort(self._weights, descending=True)
        top_samples = self._state_seq_batch[top_indices]
        top_weights = self._weights[top_indices]

        return top_samples, top_weights
