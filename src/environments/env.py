"""
author: Masafumi Endo
"""

import warnings
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import opensimplex as simplex
from typing import Optional, Tuple

from src.environments.utils import TensorData


class GridMap:
    def __init__(
        self,
        grid_size: int,
        resolution: float,
        roughness_exponent: float = 0.8,
        amplitude_gain: float = 10,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the grid map.

        Parameters:
        - grid_size (int): Number of grids along one axis.
        - resolution (float): Resolution of each grid in meters.
        - roughness_exponent (float): Roughness exponent for the fractal surface, between 0 and 1.
        - amplitude_gain (float): Amplitude gain for the fractal surface.
        - seed (Optional[int]): Seed for random number generation.
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.center_x = self.center_y = grid_size * resolution / 2
        self.lower_left_x = self.center_x - grid_size / 2 * resolution
        self.lower_left_y = self.center_y - grid_size / 2 * resolution
        self.num_grids = grid_size ** 2
        self.tensor_data = self.initialize_data(self.grid_size)
        self.roughness_exponent = roughness_exponent
        self.amplitude_gain = amplitude_gain
        self.seed = seed
        self.random_generator = self.set_randomness(seed)

    @staticmethod
    def initialize_data(grid_size: int) -> TensorData:
        """
        Initialize data structure for terrain information with zero-filled numpy arrays
        for a square grid.

        Parameters:
        - grid_size (int): Size of one side of the square grid. The grid is assumed to be square.

        Returns:
        - An instance of the Data class with initialized arrays for terrain attributes.
        """
        return TensorData(
            heights=torch.zeros((grid_size, grid_size)),
            slopes=torch.zeros((grid_size, grid_size)),
            slips=torch.zeros((grid_size, grid_size)),
            t_classes=torch.zeros((grid_size, grid_size)),
            colors=torch.zeros(
                (3, grid_size, grid_size)
            ),  # Assuming color is an RGB image
        )

    @staticmethod
    def set_randomness(seed: Optional[int] = None) -> np.random.Generator:
        """
        Set randomness for reproducibility.

        Parameters:
        - seed (Optional[int]): Seed for random number generation.

        Returns:
        - A numpy random Generator instance.
        """
        simplex.seed(seed) if seed is not None else simplex.random_seed()
        return np.random.default_rng(seed)

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
            attr_array = getattr(self.tensor_data, attribute)
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
            attr_array = getattr(self.tensor_data, attribute)

            # Increment or set the value based on the 'increment' flag
            if increment:
                attr_array[y_index, x_index] += value
            else:
                attr_array[y_index, x_index] = value

            # Update the data attribute with the modified array
            setattr(self.tensor_data, attribute, attr_array)
            return True
        else:
            return False


class TerrainGeometry:
    def __init__(self, grid_map: GridMap):
        """
        Initialize terrain geometry.

        Parameters:
        - grid_map (GridMap): An instance of GridMap.
        """
        self.grid_map = grid_map

    def set_terrain_geometry(
        self,
        is_fractal: bool = True,
        is_crater: bool = True,
        num_craters: Optional[int] = 5,
        crater_margin: Optional[float] = 5,
        min_angle: Optional[float] = 10,
        max_angle: Optional[float] = 20,
        min_radius: Optional[float] = 10,
        max_radius: Optional[float] = 20,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        safety_margin: Optional[float] = 5,
    ) -> None:
        """
        Sets the planetary terrain environment based on fractal methods with craters.

        Parameters:
        - is_fractal (bool): If fractal surface roughness should be applied.
        - is_crater (bool): If craters should be generated.
        - num_craters (int): Number of craters to generate.
        - crater_margin (float): Safety margin around craters to avoid overlap.
        - min_angle (float), max_angle (float): Min and max crater slope angles.
        - min_radius (float), max_radius (float): Min and max crater radii.
        - start_pos (Optional[torch.Tensor]), goal_pos (Optional[torch.Tensor]): Start and goal positions, to avoid craters close by.
        - safety_margin (float): Margin around start and goal positions to avoid placing craters.
        """
        # Initialize terrain height data
        heights = torch.zeros((self.grid_map.grid_size, self.grid_map.grid_size))

        # Apply crater geometry if necessary
        if is_crater:
            craters_placed = 0
            # Initialize crater positions and radii arrays
            if start_pos is not None and goal_pos is not None:
                crater_positions = torch.stack(
                    (start_pos.unsqueeze(0), goal_pos.unsqueeze(0)), dim=0
                )
            else:
                crater_positions = torch.empty((0, 2))
            crater_radii = torch.full((crater_positions.shape[0],), safety_margin)

            while craters_placed < num_craters:
                crater_center = (
                    torch.rand(2)
                    * (
                        (self.grid_map.grid_size - 1) * self.grid_map.resolution
                        - self.grid_map.lower_left_x
                    )
                    + self.grid_map.lower_left_x
                )
                crater_radius = (
                    torch.rand(1) * (max_radius - min_radius) + min_radius
                ).item()

                if not self.check_circle_overlap(
                    crater_positions,
                    crater_radii,
                    crater_center,
                    crater_radius,
                    crater_margin,
                ):
                    heights = self.generate_crater(
                        heights,
                        self.grid_map.random_generator.uniform(min_angle, max_angle),
                        crater_radius,
                        crater_center,
                    )
                    crater_positions = torch.cat(
                        (crater_positions, crater_center.unsqueeze(0)), dim=0
                    )
                    crater_radii = torch.cat(
                        (crater_radii, torch.tensor([crater_radius])), dim=0
                    )
                    craters_placed += 1

        if is_fractal:
            heights = self.generate_fractal_surface(heights)

        self.grid_map.tensor_data.heights = heights

    def generate_crater(
        self,
        heights: torch.Tensor,
        angle: float,
        radius: float,
        crater_center: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generates and applies a crater to the terrain based on the specified angle, radius,
        and center position. This version ensures the crater's slope is correctly represented
        by the specified angle.

        Parameters:
        - heights (torch.Tensor): The original height data.
        - angle (float): Inner rim angle of the crater, determining the slope of the crater walls.
        - radius (float): Radius of the crater, specifying the size of the depression.
        - crater_center (torch.Tensor): A 2D array specifying the x and y coordinates (in meters) of the crater's center.

        Returns:
        - (torch.Tensor): Height data with the crater applied.
        """
        crater_diameter = 2 * radius
        grid_resolution = self.grid_map.resolution
        grid_size = int(
            torch.ceil(
                torch.tensor(crater_diameter) / torch.tensor(grid_resolution)
            ).item()
        )
        xx, yy = torch.meshgrid(
            torch.linspace(-radius, radius, grid_size),
            torch.linspace(-radius, radius, grid_size),
            indexing="ij",
        )

        distances = torch.sqrt(xx ** 2 + yy ** 2)
        # Adjust the crater profile calculation to ensure the correct gradient
        crater_profile = torch.where(
            distances <= radius,
            -torch.tan(torch.deg2rad(torch.tensor(angle))) * (radius - distances),
            torch.zeros_like(distances),
        )
        center_x_index, center_y_index = self.grid_map.get_grid_indices_from_position(
            crater_center[0].item(), crater_center[1].item()
        )

        for i in range(grid_size):
            for j in range(grid_size):
                x_index = center_x_index - grid_size // 2 + i
                y_index = center_y_index - grid_size // 2 + j
                if (
                    0 <= x_index < self.grid_map.grid_size
                    and 0 <= y_index < self.grid_map.grid_size
                ):
                    heights[y_index, x_index] += crater_profile[
                        j, i
                    ]  # Ensure depth is subtracted to create a depression

        # Adjust the entire terrain to ensure no negative heights, if necessary
        heights = self.adjust_height_values(heights)
        return heights

    def check_circle_overlap(
        self,
        existing_centers: torch.Tensor,
        existing_radii: torch.Tensor,
        new_center: torch.Tensor,
        new_radius: float,
        margin: float,
    ) -> bool:
        """
        Checks if a new circle overlaps with existing circles using PyTorch, considering a margin.

        Parameters:
        - existing_centers (torch.Tensor): Tensor of existing circle centers.
        - existing_radii (torch.Tensor): Tensor of existing circle radii.
        - new_center (torch.Tensor): Center of the new circle.
        - new_radius (float): Radius of the new circle.
        - margin (float): Additional safety margin to consider.

        Returns:
        - (bool): True if there is an overlap, False otherwise.
        """
        distance = torch.norm(existing_centers - new_center, dim=1)
        overlap = distance < (existing_radii + new_radius + margin)
        return overlap.any().item()

    def generate_fractal_surface(self, heights: torch.Tensor) -> torch.Tensor:
        """
        Generates a fractal surface based on fractional Brownian motion (fBm),
        ensuring correct symmetry for the inverse FFT to produce real output.

        Parameters:
        - heights (torch.Tensor): The original height data.

        Returns:
        - (torch.Tensor): Generated fractal surface heights.
        """
        device = heights.device
        size = self.grid_map.grid_size
        roughness_exponent = self.grid_map.roughness_exponent
        amplitude_gain = self.grid_map.amplitude_gain
        resolution = self.grid_map.resolution

        grid = torch.zeros((size, size), dtype=torch.complex64, device=device)

        # Generate the upper left quadrant and its symmetric counterparts
        for y in range(size // 2 + 1):
            for x in range(size // 2 + 1):
                phase = 2 * torch.pi * torch.rand(1, device=device)
                if x != 0 or y != 0:  # Avoid division by zero at the origin
                    rad = torch.pow(
                        torch.tensor(
                            [x ** 2 + y ** 2], dtype=torch.float32, device=device
                        ),
                        -((roughness_exponent + 1) / 2),
                    )
                else:
                    rad = torch.tensor(0.0, device=device)
                grid[y, x] = rad * torch.exp(1j * phase)

                # Symmetry for non-edge cases
                if x > 0 and y > 0:
                    grid[-y, -x] = grid[y, x].conj()

        # Handle the edges for real parts
        edge_and_corners = [(size // 2, 0), (0, size // 2), (size // 2, size // 2)]
        for y, x in edge_and_corners:
            grid[y, x] = grid[y, x].real + 0j  # Ensuring the value remains complex

        # Adjust for the second half of the grid
        for y in range(1, size // 2):
            for x in range(1, size // 2):
                phase = 2 * torch.pi * torch.rand(1, device=device)
                rad = torch.pow(
                    torch.tensor([x ** 2 + y ** 2], dtype=torch.float32, device=device),
                    -((roughness_exponent + 1) / 2),
                )
                grid[y, size - x] = rad * torch.exp(1j * phase)
                grid[size - y, x] = grid[y, size - x].conj()

        # Scale and perform the inverse FFT
        grid *= abs(amplitude_gain) * (size * resolution * 1e3) ** (
            roughness_exponent + 1 + 0.5
        )
        surface = torch.fft.ifft2(grid).real / (resolution * 1e3) ** 2
        surface *= 1e-3  # Convert to meters

        heights += surface.to(torch.float32)
        heights = self.adjust_height_values(heights)
        return heights

    def adjust_height_values(self, heights: torch.Tensor) -> torch.Tensor:
        """
        Adjusts height values using PyTorch to ensure they start from zero.

        Parameters:
        - heights (torch.Tensor): The original height data as a PyTorch tensor.

        Returns:
        - (torch.Tensor): Adjusted height data.
        """
        return heights - heights.min()


class TerrainColoring:
    def __init__(self, grid_map: GridMap) -> None:
        self.grid_map = grid_map

    def set_terrain_class_coloring(
        self,
        occupancy: torch.Tensor,
        lower_threshold: float = 0.8,
        upper_threshold: float = 1,
        ambient_intensity: float = 0.1,
    ) -> None:
        """
        Sets the terrain distribution based on a given occupancy vector, applying shading effects
        based on the terrain class distribution.

        Parameters:
        - occupancy (torch.Tensor): Occupancy ratios for different terrain classes.
        - lower_threshold (float): Lower threshold for shading effect.
        - upper_threshold (float): Upper threshold for shading effect.
        - ambient_intensity (float): Ambient light intensity for shading.
        """
        if occupancy.sum() > 1:
            occupancy /= occupancy.sum()
            warnings.warn(
                "Sum of occupancy vector exceeds one! The vector has been normalized."
            )

        self.grid_map.tensor_data.t_classes = self.generate_multi_terrain(occupancy)
        self.create_color_map(
            occupancy=occupancy,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            ambient_intensity=ambient_intensity,
        )

    def generate_multi_terrain(
        self, occupancy: torch.Tensor, feature_size: float = 20
    ) -> torch.Tensor:
        """
        Generates terrain class data based on noise, simulating various terrain types. Segments the
        noise data into different classes based on the provided occupancy ratios.

        Parameters:
        - occupancy (torch.Tensor): Occupancy ratios for different terrain types.
        - feature_size (float): Scale of terrain features influenced by noise.

        Returns:
        - (torch.Tensor): Terrain class data as a 2D array.
        """
        # Generate noise data
        noise_data = torch.zeros((self.grid_map.grid_size, self.grid_map.grid_size))
        for y in range(self.grid_map.grid_size):
            for x in range(self.grid_map.grid_size):
                noise_data[y, x] = simplex.noise2(x / feature_size, y / feature_size)
        noise_data = (
            (noise_data - noise_data.min())
            / (noise_data.max() - noise_data.min())
            * 100
        )

        # Convert occupancy ratios into cumulative percentages for thresholding
        thresholds = torch.cumsum(occupancy, dim=0) * 100
        t_classes = torch.zeros_like(noise_data)

        # Assign classes based on thresholds
        for i, threshold in enumerate(thresholds):
            # Ensure we don't go beyond the last class
            if i < len(thresholds) - 1:  # Ensure we don't go beyond the last class
                t_classes[noise_data > threshold] = i + 1

        t_classes = F.one_hot(t_classes.long(), num_classes=occupancy.shape[0])

        return t_classes

    def create_color_map(
        self,
        occupancy: torch.Tensor,
        lower_threshold: float,
        upper_threshold: float,
        ambient_intensity: float,
    ) -> None:
        """
        Creates a color map for the terrain based on class distribution and applies shading.

        Parameters:
        - occupancy (torch.Tensor): Occupancy ratios for different terrain classes.
        - lower_threshold (float): Parameter defining the lower limit of shading effect.
        - upper_threshold (float): Parameter defining the upper limit of shading effect.
        - ambient_intensity (float): Ambient light intensity for the shading.
        """
        # Normalize terrain classes to range [-num_classes, 0] for color mapping
        num_classes = occupancy.shape[0]
        facecolors = -torch.argmax(self.grid_map.tensor_data.t_classes, dim=2)
        norm = matplotlib.colors.Normalize(vmin=-num_classes, vmax=0)
        colors = plt.cm.copper(norm(facecolors.cpu().numpy()))[:, :, 0:3]
        colors = torch.from_numpy(colors).permute(2, 0, 1)

        # Apply shading to color map
        self.create_shading(colors, lower_threshold, upper_threshold, ambient_intensity)

    def create_shading(
        self,
        colors: torch.Tensor,
        lower_threshold: float,
        upper_threshold: float = 1,
        ambient_intensity: float = 0.1,
    ) -> None:
        """
        Creates a shading effect on the terrain's color map based on the height map and specified lighting parameters.

        Parameters:
        - colors (torch.Tensor): Color map for the terrain.
        - lower_threshold (float): Lower threshold for shading effect, influencing light direction.
        - upper_threshold (float): Upper threshold for light source height.
        - ambient_intensity (float): Ambient light intensity for shading.
        """
        heights = self.grid_map.tensor_data.heights

        # calculate normal vector
        dx = heights[:, :-1] - heights[:, 1:]
        dy = heights[:-1, :] - heights[1:, :]
        norm = torch.zeros((3, self.grid_map.grid_size, self.grid_map.grid_size))
        norm[0, :, :-1] += dx
        norm[0, :, 1:] += dx
        norm[0, :, 1:-1] /= 2
        norm[1, :-1, :] += dy
        norm[1, 1:, :] += dy
        norm[1, 1:-1, :] /= 2
        norm[2, :] = 1
        norm /= torch.sqrt((norm * norm).sum(axis=0, keepdims=True))

        # generate light source vector
        light_angle = torch.rand(1) * (2 * torch.pi)
        z = torch.rand(1) * (upper_threshold - lower_threshold) + lower_threshold
        radius = torch.sqrt(1 - z ** 2)
        light_source = torch.tensor(
            [radius * torch.cos(light_angle), radius * torch.sin(light_angle), z]
        )

        # cast shading to color image
        shade = torch.sum(light_source[:, None, None] * norm, axis=0, keepdims=True)
        color_shaded = shade * colors + ambient_intensity * colors
        self.grid_map.tensor_data.colors = torch.clamp(color_shaded, 0, 1).squeeze()
