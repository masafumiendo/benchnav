"""
author: Masafumi Endo
"""

import warnings
from typing import Optional, Dict
import torch
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import opensimplex as simplex

from src.environments.grid_map import GridMap
from src.environments.slip_model import SlipModel


class TerrainGeometry:
    def __init__(
        self,
        grid_map: GridMap,
        roughness_exponent: float = 0.8,
        amplitude_gain: float = 10,
    ):
        """
        Initialize terrain geometry.

        Parameters:
        - grid_map (GridMap): An instance of GridMap.
        - roughness_exponent (float): Roughness exponent for the fractal surface, between 0 and 1.
        - amplitude_gain (float): Amplitude gain for the fractal surface.
        """
        self.grid_map = grid_map
        self.roughness_exponent = roughness_exponent
        self.amplitude_gain = amplitude_gain

    def set_terrain_geometry(
        self,
        is_fractal: bool = True,
        is_crater: bool = True,
        num_craters: Optional[int] = 3,
        crater_margin: Optional[float] = 5,
        min_angle: Optional[float] = 10,
        max_angle: Optional[float] = 20,
        min_radius: Optional[float] = 5,
        max_radius: Optional[float] = 10,
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
        # Initialize terrain height and slope tensors
        grid_size = self.grid_map.grid_size + 2  # Add padding for slope generation
        heights = torch.zeros((grid_size, grid_size), device=self.grid_map.device)

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

            # Place craters
            count = 0
            while craters_placed < num_craters:
                crater_center = (
                    torch.rand(2)
                    * (
                        (grid_size - 1) * self.grid_map.resolution
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
                        torch.rand(1).item() * (max_angle - min_angle) + min_angle,
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

                count += 1
                if count > 1000:
                    warnings.warn(
                        "Failed to place all craters after 1000 attempts. Consider adjusting the parameters."
                    )
                    break

        if is_fractal:
            heights = self.generate_fractal_surface(heights)

        # Crop the edges to avoid nan values for slope generation
        self.grid_map.tensor_data["heights"] = heights[1:-1, 1:-1]
        self.grid_map.tensor_data["slopes"] = self.generate_slopes(heights)

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
                    0 <= x_index < self.grid_map.grid_size + 2
                    and 0 <= y_index < self.grid_map.grid_size + 2
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
        device = self.grid_map.device
        grid_size = self.grid_map.grid_size + 2  # Add padding for slope generation
        roughness_exponent = self.roughness_exponent
        amplitude_gain = self.amplitude_gain
        resolution = self.grid_map.resolution

        grid = torch.zeros((grid_size, grid_size), dtype=torch.complex64, device=device)

        # Generate the upper left quadrant and its symmetric counterparts
        for y in range(grid_size // 2 + 1):
            for x in range(grid_size // 2 + 1):
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
        edge_and_corners = [
            (grid_size // 2, 0),
            (0, grid_size // 2),
            (grid_size // 2, grid_size // 2),
        ]
        for y, x in edge_and_corners:
            grid[y, x] = grid[y, x].real + 0j  # Ensuring the value remains complex

        # Adjust for the second half of the grid
        for y in range(1, grid_size // 2):
            for x in range(1, grid_size // 2):
                phase = 2 * torch.pi * torch.rand(1, device=device)
                rad = torch.pow(
                    torch.tensor([x ** 2 + y ** 2], dtype=torch.float32, device=device),
                    -((roughness_exponent + 1) / 2),
                )
                grid[y, grid_size - x] = rad * torch.exp(1j * phase)
                grid[grid_size - y, x] = grid[y, grid_size - x].conj()

        # Scale and perform the inverse FFT
        grid *= abs(amplitude_gain) * (grid_size * resolution * 1e3) ** (
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

    def generate_slopes(self, heights: torch.Tensor) -> torch.Tensor:
        """
        Generates slope values based on the given height data by following Horn's method.

        Parameters:
        - heights (torch.Tensor): The original height data.

        Returns:
        - (torch.Tensor): Generated slope data in degrees. Note that the edges are assigned inf.
        """
        sobel_x = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
            dtype=torch.float32,
            device=self.grid_map.device,
        )
        sobel_y = torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
            dtype=torch.float32,
            device=self.grid_map.device,
        )

        heights = heights.unsqueeze(0).unsqueeze(0)

        gradient_x = F.conv2d(heights, sobel_x, padding=1) / (
            8 * self.grid_map.resolution
        )
        gradient_y = F.conv2d(heights, sobel_y, padding=1) / (
            8 * self.grid_map.resolution
        )

        slopes = torch.atan(torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2)))
        slopes = torch.rad2deg(slopes).squeeze()

        # Crop the edges to avoid inf values
        slopes = slopes[1:-1, 1:-1]
        return slopes


class TerrainColoring:
    def __init__(self, grid_map: GridMap) -> None:
        """
        Initialize terrain coloring.

        Parameters:
        - grid_map (GridMap): An instance of GridMap.
        """
        self.grid_map = grid_map

    def set_terrain_class_coloring(
        self,
        occupancy: Optional[torch.Tensor] = None,
        lower_threshold: float = 0.8,
        upper_threshold: float = 1,
        ambient_intensity: float = 0.1,
    ) -> None:
        """
        Sets the terrain distribution based on a given occupancy vector, applying shading effects
        based on the terrain class distribution.

        Parameters:
        - occupancy (Optional[torch.Tensor]): Occupancy ratios for different terrain classes.
        - lower_threshold (float): Lower threshold for shading effect.
        - upper_threshold (float): Upper threshold for shading effect.
        - ambient_intensity (float): Ambient light intensity for shading.
        """
        if occupancy is None:
            occupancy = torch.tensor([1.0])

        if occupancy.sum() > 1:
            occupancy /= occupancy.sum()
            warnings.warn(
                "Sum of occupancy vector exceeds one! The vector has been normalized."
            )

        self.grid_map.tensor_data["t_classes"] = self.generate_multi_terrain(occupancy)
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
        noise_data = torch.zeros(
            (self.grid_map.grid_size, self.grid_map.grid_size),
            device=self.grid_map.device,
        )

        for y in range(self.grid_map.grid_size):
            for x in range(self.grid_map.grid_size):
                noise_data[y, x] = simplex.noise2(x / feature_size, y / feature_size)
        noise_data = (
            (noise_data - noise_data.min())
            / (noise_data.max() - noise_data.min())
            * 100
        )

        # Generate thresholds for terrain classes
        thresholds = torch.cumsum(occupancy, dim=0) * 100
        # Assign classes based on thresholds and noise data values (0-100)
        t_classes = torch.full_like(noise_data, fill_value=-1, dtype=torch.long)
        start_index = (occupancy > 0).nonzero().min().item()  # Start index for classes
        for i, threshold in enumerate(thresholds[start_index:], start=start_index):
            if i == start_index:
                mask = noise_data <= threshold
            else:
                mask = (noise_data > thresholds[i - 1]) & (noise_data <= threshold)
            t_classes[mask] = i

        # If -1 exists, raise a warning
        if (t_classes == -1).any():
            warnings.warn("Some grid cells have not been assigned a terrain class.")

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
        facecolors = self.grid_map.tensor_data["t_classes"]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=num_classes - 1)
        colors = plt.cm.copper(norm(facecolors.cpu().numpy()))[:, :, 0:3]
        colors = (
            torch.from_numpy(colors)
            .permute(2, 0, 1)
            .to(device=self.grid_map.device, dtype=torch.float32)
        )

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
        heights = self.grid_map.tensor_data["heights"]

        # Calculate normal vector
        dx = heights[:, :-1] - heights[:, 1:]
        dy = heights[:-1, :] - heights[1:, :]
        norm = torch.zeros(
            (3, self.grid_map.grid_size, self.grid_map.grid_size),
            dtype=torch.float32,
            device=self.grid_map.device,
        )
        norm[0, :, :-1] += dx
        norm[0, :, 1:] += dx
        norm[0, :, 1:-1] /= 2
        norm[1, :-1, :] += dy
        norm[1, 1:, :] += dy
        norm[1, 1:-1, :] /= 2
        norm[2, :] = 1
        norm /= torch.sqrt((norm * norm).sum(axis=0, keepdims=True))

        # Generate light source vector
        light_angle = torch.rand(1) * (2 * torch.pi)
        z = torch.rand(1) * (upper_threshold - lower_threshold) + lower_threshold
        radius = torch.sqrt(1 - z ** 2)
        light_source = torch.tensor(
            [radius * torch.cos(light_angle), radius * torch.sin(light_angle), z],
            device=self.grid_map.device,
        )

        # Cast shading to color image
        shade = torch.sum(light_source[:, None, None] * norm, axis=0, keepdims=True)
        color_shaded = shade * colors + ambient_intensity * colors
        color_shaded = torch.clamp(color_shaded, 0, 1).squeeze()
        self.grid_map.tensor_data["colors"] = torch.clamp(color_shaded, 0, 1).squeeze()


class TerrainTraversability:
    def __init__(self, grid_map: GridMap) -> None:
        """
        Initialize terrain traversability.

        Parameters:
        - grid_map (GridMap): An instance of GridMap.
        """

        self.grid_map = grid_map
        # Check tensor data for terrain information
        if "slopes" not in self.grid_map.tensor_data:
            raise ValueError("Slope data is required for setting traversability.")
        if "t_classes" not in self.grid_map.tensor_data:
            raise ValueError(
                "Terrain class data is required for setting traversability."
            )

    def set_traversability(self, slip_models: Dict[int, SlipModel]) -> None:
        """
        Sets the traversability of the terrain based on the slip models provided.

        Parameters:
        - slip_models (Dict[int, SlipModel]): Slip models for each terrain class.
        """
        t_classes = self.grid_map.tensor_data["t_classes"]

        # Check if the number of terrain classes exceeds the number of slip models
        if t_classes.unique().min() < 0 or t_classes.unique().max() >= len(slip_models):
            raise ValueError(
                "The number of terrain classes exceeds the number of slip models."
            )

        slips_mean = torch.full(
            (self.grid_map.grid_size, self.grid_map.grid_size),
            torch.inf,
            device=self.grid_map.device,
        )
        slips_stddev = torch.full(
            (self.grid_map.grid_size, self.grid_map.grid_size),
            torch.inf,
            device=self.grid_map.device,
        )

        slopes = self.grid_map.tensor_data["slopes"]
        for t_class, slip_model in slip_models.items():
            mask = t_classes == t_class
            if mask.any():  # Check if there are any elements in this class
                masked_slopes = slopes[mask]
                distribution = slip_model.model_distribution(masked_slopes)
                slips_mean[mask] = distribution.mean
                slips_stddev[mask] = distribution.stddev

        self.grid_map.tensor_data["slips"] = Normal(slips_mean, slips_stddev)
