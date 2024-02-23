"""
author: Masafumi Endo
"""

import warnings
import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
import opensimplex as simplex
from typing import Optional, Tuple

from environments.utils import Data


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
        self.num_grids = grid_size**2
        self.data = self.initialize_data(self.grid_size)
        self.roughness_exponent = roughness_exponent
        self.amplitude_gain = amplitude_gain
        self.seed = seed
        self.random_generator = self.set_randomness(seed)

    @staticmethod
    def initialize_data(grid_size: int) -> Data:
        """
        Initialize data structure for terrain information with zero-filled numpy arrays
        for a square grid.

        Parameters:
        - grid_size (int): Size of one side of the square grid. The grid is assumed to be square.

        Returns:
        - An instance of the Data class with initialized arrays for terrain attributes.
        """
        return Data(
            height=np.zeros((grid_size, grid_size)),
            slope=np.zeros((grid_size, grid_size)),
            slip=np.zeros((grid_size, grid_size)),
            t_class=np.zeros((grid_size, grid_size)),
            color=np.zeros((grid_size, grid_size, 3)),  # Assuming color is an RGB image
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
        self, x_index: int, y_index: int, attribute: str = "height"
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
            attr_array = getattr(self.data, attribute)
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
        self, x_pos: float, y_pos: float, value: float, attribute: str = "height"
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
        attribute: str = "height",
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
            attr_array = getattr(self.data, attribute)

            # Increment or set the value based on the 'increment' flag
            if increment:
                attr_array[y_index, x_index] += value
            else:
                attr_array[y_index, x_index] = value

            # Update the data attribute with the modified array
            setattr(self.data, attribute, attr_array)
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
        num_craters: int = 5,
        crater_margin: float = 5,
        min_angle: float = 10,
        max_angle: float = 20,
        min_radius: float = 10,
        max_radius: float = 20,
        start_pos: Optional[NDArray] = None,
        goal_pos: Optional[NDArray] = None,
        safety_margin: float = 5,
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
        - start_pos (Optional[NDArray]), goal_pos (Optional[NDArray]): Start and goal positions, to avoid craters close by.
        - safety_margin (float): Margin around start and goal positions to avoid placing craters.
        """
        if is_crater:
            craters_placed = 0
            # Initialize crater positions and radii arrays
            crater_positions = (
                np.concatenate([np.atleast_2d(start_pos), np.atleast_2d(goal_pos)])
                if start_pos is not None and goal_pos is not None
                else np.empty((0, 2))
            )
            crater_radii = np.full((crater_positions.shape[0],), safety_margin)

            while craters_placed < num_craters:
                crater_center = self.grid_map.random_generator.uniform(
                    self.grid_map.lower_left_x,
                    (self.grid_map.grid_size - 1) * self.grid_map.resolution,
                    size=(2,),
                )
                crater_radius = self.grid_map.random_generator.uniform(
                    min_radius, max_radius
                )

                if not self.check_circle_overlap(
                    crater_positions,
                    crater_radii,
                    crater_center,
                    crater_radius,
                    crater_margin,
                ):
                    self.generate_crater(
                        self.grid_map.random_generator.uniform(min_angle, max_angle),
                        crater_radius,
                        crater_center,
                    )
                    crater_positions = np.vstack([crater_positions, crater_center])
                    crater_radii = np.append(crater_radii, crater_radius)
                    craters_placed += 1

        if is_fractal:
            self.apply_fractal_surface()

        self.grid_map.data.height = self.adjust_height_values(self.grid_map.data.height)

    def apply_fractal_surface(self) -> None:
        """
        Applies a fractal surface to the terrain height data.
        """
        fractal_heights = self.generate_fractal_surface()
        self.grid_map.data.height += fractal_heights

    def generate_fractal_surface(self) -> NDArray:
        """
        Generates a fractal surface based on fractional Brownian motion (fBm),
        ensuring correct symmetry for the inverse FFT to produce real output.

        Returns:
        - (NDArray): Generated fractal surface heights.
        """
        size = self.grid_map.grid_size
        roughness_exponent = self.grid_map.roughness_exponent
        amplitude_gain = self.grid_map.amplitude_gain
        resolution = self.grid_map.resolution

        grid = np.zeros((size, size), dtype=np.complex64)

        # Generate the upper left quadrant and its symmetric counterparts
        for y in range(size // 2 + 1):
            for x in range(size // 2 + 1):
                phase = 2 * np.pi * self.grid_map.random_generator.random()
                if x != 0 or y != 0:  # Avoid division by zero at the origin
                    rad = 1 / (x**2 + y**2) ** ((roughness_exponent + 1) / 2)
                else:
                    rad = 0.0
                grid[y, x] = rad * np.exp(1j * phase)

                # Symmetry for non-edge cases
                if x > 0 and y > 0:
                    grid[-y, -x] = np.conj(grid[y, x])

        # Handle the edges for real parts
        grid[size // 2, 0] = np.real(grid[size // 2, 0])
        grid[0, size // 2] = np.real(grid[0, size // 2])
        grid[size // 2, size // 2] = np.real(grid[size // 2, size // 2])

        # Adjust for the second half of the grid
        for y in range(1, size // 2):
            for x in range(1, size // 2):
                phase = 2 * np.pi * self.grid_map.random_generator.random()
                rad = 1 / (x**2 + y**2) ** ((roughness_exponent + 1) / 2)
                grid[y, size - x] = rad * np.exp(1j * phase)
                grid[size - y, x] = np.conj(grid[y, size - x])

        # Scale and perform the inverse FFT
        grid *= abs(amplitude_gain) * (size * resolution * 1e3) ** (
            roughness_exponent + 1 + 0.5
        )
        surface = np.fft.ifft2(grid).real / (resolution * 1e3) ** 2
        surface *= 1e-3  # Convert to meters

        return surface.astype(np.float32)

    def generate_crater(
        self, angle: float, radius: float, crater_center: NDArray
    ) -> None:
        """
        Generates and applies a crater to the terrain based on the specified angle, radius,
        and center position. This version ensures the crater's slope is correctly represented
        by the specified angle.

        Parameters:
        - angle (float): Inner rim angle of the crater, determining the slope of the crater walls.
        - radius (float): Radius of the crater, specifying the size of the depression.
        - crater_center (NDArray): A 2D array specifying the x and y coordinates (in meters) of the crater's center.
        """
        crater_diameter = 2 * radius
        grid_resolution = self.grid_map.resolution
        grid_size = int(np.ceil(crater_diameter / grid_resolution))
        xx, yy = np.meshgrid(
            np.linspace(-radius, radius, grid_size),
            np.linspace(-radius, radius, grid_size),
        )

        distances = np.sqrt(xx**2 + yy**2)
        # Adjust the crater profile calculation to ensure the correct gradient
        crater_profile = np.where(
            distances <= radius, -np.tan(np.radians(angle)) * (radius - distances), 0
        )
        center_x_index, center_y_index = self.grid_map.get_grid_indices_from_position(
            crater_center[0], crater_center[1]
        )

        for i in range(grid_size):
            for j in range(grid_size):
                x_index = center_x_index - grid_size // 2 + i
                y_index = center_y_index - grid_size // 2 + j
                if (
                    0 <= x_index < self.grid_map.grid_size
                    and 0 <= y_index < self.grid_map.grid_size
                ):
                    self.grid_map.data.height[y_index, x_index] += crater_profile[
                        j, i
                    ]  # Ensure depth is subtracted to create a depression

        # Adjust the entire terrain to ensure no negative heights, if necessary
        self.grid_map.data.height = self.adjust_height_values(self.grid_map.data.height)

    def check_circle_overlap(
        self,
        existing_centers: NDArray,
        existing_radii: NDArray,
        new_center: NDArray,
        new_radius: float,
        margin: float,
    ) -> bool:
        """
        Checks if a new circle overlaps with existing circles, considering a margin.

        Parameters:
        - existing_centers (NDArray): Array of existing circle centers.
        - existing_radii (NDArray): Array of existing circle radii.
        - new_center (NDArray): Center of the new circle.
        - new_radius (float): Radius of the new circle.
        - margin (float): Additional safety margin to consider.

        Returns:
        - (bool): True if there is an overlap, False otherwise.
        """
        for center, radius in zip(existing_centers, existing_radii):
            distance = np.linalg.norm(center - new_center)
            if distance < radius + new_radius + margin:
                return True
        return False

    def adjust_height_values(self, heights: NDArray) -> NDArray:
        """
        Adjusts height values to ensure they start from zero.

        Parameters:
        - heights (NDArray): The original height data.

        Returns:
        - (NDArray): Adjusted height data.
        """
        return heights - np.min(heights)


class TerrainColoring:
    def __init__(self, grid_map: GridMap) -> None:
        self.grid_map = grid_map

    def set_terrain_class_distribution(
        self,
        occupancy: list,
        threshold: float = 0.8,
        upper_threshold: float = 1,
        ambient_intensity: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Sets the terrain distribution based on a given occupancy vector, applying shading effects
        based on the terrain class distribution.

        Parameters:
        - occupancy (list): List of occupancy ratios for different terrain classes.
        - threshold (float): Lower threshold for shading effect.
        - upper_threshold (float): Upper threshold for shading effect.
        - ambient_intensity (float): Ambient light intensity for shading.
        - seed (Optional[int]): Seed for random number generation, affecting terrain noise.
        """
        self.occupancy = np.array(occupancy)
        if self.occupancy.sum() > 1:
            self.occupancy /= self.occupancy.sum()
            warnings.warn(
                "Sum of occupancy vector exceeds one! The vector has been normalized."
            )

        terrain_data = self.generate_multi_terrain(seed=seed)
        self.grid_map.data.t_class = terrain_data.ravel()
        self.create_color_map(
            occupancy=occupancy,
            threshold=threshold,
            upper_threshold=upper_threshold,
            ambient_intensity=ambient_intensity,
        )

    def generate_multi_terrain(
        self, feature_size: float = 20, seed: Optional[int] = None
    ) -> NDArray:
        """
        Generates terrain class data based on noise, simulating various terrain types. Segments the
        noise data into different classes based on the provided occupancy ratios.

        Parameters:
        - feature_size (float): Scale of terrain features influenced by noise.
        - seed (Optional[int]): Seed for random number generation, affecting noise generation.

        Returns:
        - (NDArray): Terrain class data as a 2D array.
        """
        if seed is not None:
            np.random.seed(seed)
            simplex.seed(seed)

        # Generate noise data
        noise_data = np.zeros((self.grid_map.grid_size, self.grid_map.grid_size))
        for y in range(self.grid_map.grid_size):
            for x in range(self.grid_map.grid_size):
                noise_data[y, x] = simplex.noise2(x / feature_size, y / feature_size)
        noise_data = (
            (noise_data - noise_data.min())
            / (noise_data.max() - noise_data.min())
            * 100
        )

        # Convert occupancy ratios into cumulative percentages for thresholding
        thresholds = np.cumsum(self.occupancy) * 100
        terrain_data = np.zeros_like(noise_data)

        # Initial class assignment to the lowest class
        terrain_data.fill(0)  # Assuming class '0' as the default/first class

        # Assign classes based on thresholds
        for i, threshold in enumerate(thresholds):
            # Cells with noise values greater than the current threshold are assigned the next class
            if i < len(thresholds) - 1:  # Ensure we don't go beyond the last class
                terrain_data[noise_data > threshold] = i + 1

        return terrain_data

    def create_color_map(
        self,
        occupancy: list,
        threshold: float,
        upper_threshold: float,
        ambient_intensity: float,
    ) -> None:
        """
        Creates a color map for the terrain based on class distribution and applies shading.

        Parameters:
        - threshold (float): Parameter influencing the start of shading effect.
        - upper_threshold (float): Parameter defining the upper limit of shading effect.
        - ambient_intensity (float): Ambient light intensity for the shading.
        """
        # Normalize terrain classes to range [-num_classes, 0] for color mapping
        num_classes = len(occupancy)
        facecolors = -np.reshape(
            self.grid_map.data.t_class,
            (self.grid_map.grid_size, self.grid_map.grid_size),
        )
        norm = matplotlib.colors.Normalize(vmin=-num_classes, vmax=0)
        self.grid_map.data.color = plt.cm.copper(norm(facecolors))[:, :, 0:3].astype(
            np.float32
        )

        # Apply shading to color map
        self.create_shading(threshold, upper_threshold, ambient_intensity)

    def create_shading(
        self,
        threshold: float,
        upper_threshold: float = 1,
        ambient_intensity: float = 0.1,
    ) -> None:
        """
        Creates a shading effect on the terrain's color map based on the height map and specified lighting parameters.

        Parameters:
        - threshold (float): Lower threshold for shading effect, influencing light direction.
        - upper_threshold (float): Upper threshold for light source height.
        - ambient_intensity (float): Ambient light intensity for shading.
        """
        height = np.reshape(
            self.grid_map.data.height,
            (self.grid_map.grid_size, self.grid_map.grid_size),
        )
        color = self.grid_map.data.color.transpose(2, 0, 1).astype(np.float32)

        # calculate normal vector
        dx = height[:, :-1] - height[:, 1:]
        dy = height[:-1, :] - height[1:, :]
        norm = np.zeros((3, self.grid_map.grid_size, self.grid_map.grid_size))
        norm[0, :, :-1] += dx
        norm[0, :, 1:] += dx
        norm[0, :, 1:-1] /= 2
        norm[1, :-1, :] += dy
        norm[1, 1:, :] += dy
        norm[1, 1:-1, :] /= 2
        norm[2, :] = 1
        norm /= np.sqrt((norm * norm).sum(axis=0, keepdims=True))

        # generate light source vector
        light_angle = self.grid_map.random_generator.random() * (2 * np.pi)
        z = (
            self.grid_map.random_generator.random() * (upper_threshold - threshold)
            + threshold
        )
        radius = np.sqrt(1 - z**2)
        light_source = np.array(
            [radius * np.cos(light_angle), radius * np.sin(light_angle), z]
        )

        # cast shading to color image
        shade = np.sum(
            light_source[:, np.newaxis, np.newaxis] * norm, axis=0, keepdims=True
        )
        color_shaded = shade * color + ambient_intensity * color
        self.grid_map.data.color = np.squeeze(np.clip(color_shaded, 0, 1)).transpose(
            1, 2, 0
        )
