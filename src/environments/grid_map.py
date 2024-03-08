"""
author: Masafumi Endo
"""

import numpy as np
import torch
from torch.distributions import Normal
from typing import Optional, Tuple

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
        - tensor_data (Optional[dict[torch.Tensor]]): Data structure for terrain information.
        - seed (Optional[int]): Seed for random number generation.
        - device (Optional[str]): Device to run the model on.
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.center_x = self.center_y = grid_size * resolution / 2
        self.lower_left_x = self.center_x - grid_size / 2 * resolution
        self.lower_left_y = self.center_y - grid_size / 2 * resolution
        self.num_grids = grid_size ** 2
        set_randomness(seed) if seed is not None else None
        # Initialize data structure for terrain information
        self.tensor_data = (
            tensor_data if tensor_data is not None else self.initialize_data(grid_size)
        )
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    @staticmethod
    def initialize_data(grid_size: int) -> dict:
        """
        Initialize data structure for terrain information with zero-filled numpy arrays
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
        - An instance of the Data class with initialized arrays for terrain attributes.
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

    def get_value_from_grid_id(
        self, x_index: int, y_index: int, attribute: str = "heights"
    ) -> Optional[float]:
        """
        Get values at a specified location described by x and y indices from the data structure.

        Parameters:
        - x_index (int): X index.
        - y_index (int): Y index.
        - attribute (str): Name of the attribute in the data structure.

        Returns:
        - Value at the specified grid location or None if out of bounds.
        """
        if 0 <= x_index < self.grid_size and 0 <= y_index < self.grid_size:
            attr_array = self.tensor_data[attribute]
            return attr_array[y_index, x_index]
        else:
            return None

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
            x_pos, self.lower_left_x, self.grid_size
        )
        y_index = self.calculate_index_from_position(
            y_pos, self.lower_left_y, self.grid_size
        )
        return x_index, y_index

    def calculate_grid_id(self, x_index: int, y_index: int) -> int:
        """
        Calculate one-dimensional grid index from x and y indices (2D to 1D transformation).

        Parameters:
        - x_index (int): X index.
        - y_index (int): Y index.

        Returns:
        - Computed grid ID as an integer.
        """
        return y_index * self.grid_size + x_index

    def calculate_index_from_position(
        self, position: float, lower_left: float, max_index: int
    ) -> int:
        """
        Calculate x or y axis index for given positional information.

        Parameters:
        - position (float): X or Y axis position.
        - lower_left (float): Lower left information.
        - max_index (int): Maximum index (width or height of the grid).

        Returns:
        - Calculated index as an integer.
        """
        index = int(np.floor((position - lower_left) / self.resolution))
        if not 0 <= index < max_index:
            raise ValueError("Given position is out of the map!")
        return index

    def set_value_from_position(
        self, x_pos: float, y_pos: float, value: float, attribute: str = "heights"
    ) -> bool:
        """
        Substitute given values into the data structure at specified x and y position.

        Parameters:
        - x_pos (float): X position in meters.
        - y_pos (float): Y position in meters.
        - value (float): Value to set.
        - attribute (str): Data attribute name to modify.

        Returns:
        - True if the value was successfully set, False otherwise.
        """
        x_index, y_index = self.get_grid_indices_from_position(x_pos, y_pos)
        return self.set_value_from_indices(x_index, y_index, value, attribute)

    def set_value_from_indices(
        self,
        x_index: int,
        y_index: int,
        value: float,
        attribute: str = "heights",
        increment: bool = True,
    ) -> bool:
        """
        Substitute given values into the data structure at specified x and y indices.

        Parameters:
        - x_index (int): X index.
        - y_index (int): Y index.
        - value (float): Value to set or increment.
        - attribute (str): Data attribute name to modify.
        - increment (bool): If True, increment existing value. Otherwise, overwrite.

        Returns:
        - True if the operation was successful, False otherwise.
        """
        if 0 <= x_index < self.grid_size and 0 <= y_index < self.grid_size:
            # Get the attribute array using getattr
            attr_array = self.tensor_data[attribute]

            # Increment or set the value based on the 'increment' flag
            if increment:
                attr_array[y_index, x_index] += value
            else:
                attr_array[y_index, x_index] = value

            return True
        else:
            return False
