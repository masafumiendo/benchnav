"""
Masafumi Endo, 2024
"""

import torch
from torch.distributions import Normal
from typing import Optional, Tuple, Dict

from src.utils.utils import set_randomness


class GridMap:
    def __init__(
        self,
        grid_size: int,
        resolution: float,
        seed: Optional[int] = None,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        distributions: Optional[dict[str, Normal]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the grid map.

        Parameters:
        - grid_size (int): Number of grids along one axis.
        - resolution (float): Resolution of each grid in meters.
        - seed (Optional[int]): Seed for random number generation. Note that this is primarily used for terrain properties generation.
        - tensors (Optional[dict[str, torch.Tensor]]): Data structure for distinct terrain information.
        - distributions (Optional[dict[str, Normal]]): Distributions for terrain information.
        - device (Optional[str]): Device to run the model on.
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.center_x = self.center_y = grid_size * resolution / 2
        self.x_limits = (
            self.center_x - grid_size / 2 * resolution,
            self.center_x + grid_size / 2 * resolution,
        )
        self.y_limits = (
            self.center_y - grid_size / 2 * resolution,
            self.center_y + grid_size / 2 * resolution,
        )
        self.num_grids = grid_size ** 2
        set_randomness(seed) if seed is not None else None
        # Initialize data structure for terrain information
        self.tensors, self.distributions = self.initialize_terrain_data(
            grid_size, tensors, distributions
        )
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    @staticmethod
    def initialize_terrain_data(
        grid_size: int,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        distributions: Optional[dict[str, Normal]] = None,
    ) -> Tuple[dict[str, torch.Tensor], dict[str, Normal]]:
        """
        Initialize data structure for terrain information with zero-filled torch tensors
        for a square grid.

        Attributes:
        - tensors (dict[str, torch.Tensor]): Dictionary of torch tensors representing terrain information.
            - heights (torch.Tensor): Tensor of shape [grid_size, grid_size] representing terrain heights.
            - slopes (torch.Tensor): Tensor of shape [grid_size, grid_size] representing terrain slopes.
            - t_classes (torch.Tensor): Tensor of shape [grid_size, grid_size] representing terrain classes.
            - colors (torch.Tensor): Tensor of shape [3, grid_size, grid_size] representing terrain colors.
        - distributions (dict[str, Normal]): Dictionary of distributions representing terrain information.
            - ground_truths (Normal): Normal distribution representing latent slip model.
            - predictions (Normal): Normal distribution representing slip predictions.
        
        Parameters:
        - grid_size (int): Size of one side of the square grid. The grid is assumed to be square.
        - tensors (Optional[dict[str, torch.Tensor]]): Data structure for distinct terrain information.
        - distributions (Optional[dict[str, Normal]]): Distributions for terrain information.

        Returns:
        - A dictionary of torch tensors and distributions representing terrain information.
        """
        tensors = (
            {
                "heights": torch.zeros((grid_size, grid_size)),
                "slopes": torch.zeros((grid_size, grid_size)),
                "t_classes": torch.zeros((grid_size, grid_size)),
                "colors": torch.zeros((3, grid_size, grid_size)),
            }
            if tensors is None
            else tensors
        )
        distributions = (
            {
                "latent_models": Normal(
                    torch.zeros((grid_size, grid_size)),
                    torch.ones((grid_size, grid_size)),
                ),
                "predictions": Normal(
                    torch.zeros((grid_size, grid_size)),
                    torch.ones((grid_size, grid_size)),
                ),
            }
            if distributions is None
            else distributions
        )
        return tensors, distributions

    def get_values_from_positions(self, positions: torch.Tensor, attribute: str):
        """
        Get corresponding value information or a masked distribution for batches of position data across multiple trajectories,
        based on a specified attribute.

        Parameters:
        - positions (torch.Tensor): A tensor of shape [batch_size, num_positions, 3] where each row corresponds to (x, y, theta) for each state.
        - attribute (str): A string specifying which attribute to retrieve (e.g., 'heights', 'slopes', 'latent_models').

        Returns:
        - A tensor containing the requested value information or a masked distribution for the states across all trajectories, based on the specified attribute.
        """
        indices = self.get_grid_indices_from_positions(positions)
        batch_size, num_positions = indices.shape[:2]

        # Flatten indices to use advanced indexing, handle 3D -> 2D conversion
        flat_indices = indices.view(-1, 2)

        if attribute in self.tensors:
            # Fetching from tensors
            fetched_values = self.tensors[attribute][
                flat_indices[:, 0], flat_indices[:, 1]
            ].view(batch_size, num_positions)
            return fetched_values
        elif attribute in self.distributions:
            # Handling distributions: create a masked distribution based on indices.
            distribution = self.distributions[attribute]
            means = distribution.mean[flat_indices[:, 0], flat_indices[:, 1]].view(
                batch_size, num_positions
            )
            stddevs = distribution.stddev[flat_indices[:, 0], flat_indices[:, 1]].view(
                batch_size, num_positions
            )
            return Normal(means, stddevs)
        else:
            raise KeyError(
                f"Attribute '{attribute}' not found in tensors or distributions."
            )

    def get_grid_indices_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get grid indices for batches of positional information for multiple trajectories.

        Parameters:
        - positions (torch.Tensor): A tensor of shape [batch_size, num_positions, 3] 
          with each row being (x_pos, y_pos, theta) for each state in each trajectory.

        Returns:
        - A tensor of shape [batch_size, num_positions, 2] with each row being (x_index, y_index).
        """
        # Convert positions to grid indices, accounting for 3D structure
        indices = (
            (
                (
                    positions[..., :2]
                    - torch.tensor(
                        [self.x_limits[0], self.y_limits[0]], device=self.device
                    )
                )
                / self.resolution
            )
            .floor()
            .int()
        )
        # Ensure indices are within bounds
        indices = indices.clamp(0, self.grid_size - 1)
        return indices
