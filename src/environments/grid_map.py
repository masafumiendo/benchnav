"""
Masafumi Endo, 2024
"""

import numpy as np
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
        tensor_data: Optional[dict[torch.Tensor]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the grid map.

        Parameters:
        - grid_size (int): Number of grids along one axis.
        - resolution (float): Resolution of each grid in meters.
        - seed (Optional[int]): Seed for random number generation. Note that this is primarily used for terrain properties generation.
        - tensor_data (Optional[dict[torch.Tensor]]): Data structure for terrain information.
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
        self.tensor_data = (
            tensor_data
            if tensor_data is not None
            else self.initialize_tensor_data(grid_size)
        )
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    @staticmethod
    def initialize_tensor_data(grid_size: int) -> Dict[str, torch.Tensor]:
        """
        Initialize data structure for terrain information with zero-filled torch tensors
        for a square grid.

        Attributes:
        - heights (torch.Tensor): A 2D tensor representing the terrain height map.
        - slopes (torch.Tensor): A 2D tensor representing the terrain slope values.
        - t_classes (torch.Tensor): A 2D tensor representing terrain classification, where each class is indicated by one-hot encoding.
        - colors (torch.Tensor): A 3D tensor (3 x height x width) representing the RGB color values for visualizing the terrain.
        - slips (Normal): A 2D tensor representing the slip values at each grid location.
        
        Parameters:
        - grid_size (int): Size of one side of the square grid. The grid is assumed to be square.

        Returns:
        - A dictionary of torch tensors and a normal distribution representing terrain information.
        """
        return {
            "heights": torch.zeros((grid_size, grid_size)),
            "slopes": torch.zeros((grid_size, grid_size)),
            "t_classes": torch.zeros((grid_size, grid_size)),
            "colors": torch.zeros((3, grid_size, grid_size)),
            "slips": Normal(
                torch.zeros((grid_size, grid_size)), torch.ones((grid_size, grid_size))
            ),
        }

    def get_grid_indices_from_position(
        self, x_pos: float, y_pos: float
    ) -> Tuple[int, int]:
        """
        Get grid indices for given positional information.

        Parameters:
        - x_pos (float): X position in meters.
        - y_pos (float): Y position in meters.

        Returns:
        - A tuple of (x_index, y_index).
        """
        x_index = self.calculate_index_from_position(
            x_pos, self.x_limits[0], self.grid_size
        )
        y_index = self.calculate_index_from_position(
            y_pos, self.y_limits[0], self.grid_size
        )
        return x_index, y_index

    def calculate_index_from_position(
        self, position: float, lower_limit: float, max_index: int
    ) -> int:
        """
        Calculate x or y axis index for given positional information.

        Parameters:
        - position (float): X or Y axis position.
        - lower_limit (float): Lower limit of the axis.
        - max_index (int): Maximum index (width or height of the grid).

        Returns:
        - Calculated index as an integer.
        """
        index = int(np.floor((position - lower_limit) / self.resolution))
        if not 0 <= index < max_index:
            raise ValueError("Given position is out of the map!")
        return index