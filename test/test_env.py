"""
Masafumi Endo, 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from environments.grid_map import GridMap
from environments.terrain_properties import TerrainGeometry
from environments.terrain_properties import TerrainColoring

# Initialize GridMap
grid_size = 64
resolution = 0.5
seed = 1

grid_map = GridMap(grid_size, resolution, seed)

# Set Terrain Geometry
terrain_geometry = TerrainGeometry(grid_map)
terrain_geometry.set_terrain_geometry(
    is_fractal=True,
    is_crater=True,
    num_craters=3,
    crater_margin=5,
    min_angle=20,
    max_angle=25,
    min_radius=5,
    max_radius=10,
)

# Set Terrain Coloring
terrain_coloring = TerrainColoring(grid_map)
occupancy = torch.tensor(
    [0.4, 0.1, 0.4, 0.1]
)  # Example occupancy ratios for terrain types
lower_threshold = 0.8
upper_threshold = 1
ambient_intensity = 0.1
terrain_coloring.set_terrain_class_coloring(
    occupancy, lower_threshold, upper_threshold, ambient_intensity
)

# Visualization
fig, axs = plt.subplots(1, 4, figsize=(18, 6))

# Height Map
axs[0].imshow(grid_map.tensors["heights"].cpu().numpy(), cmap="turbo")
axs[0].set_title("Height Map")
axs[0].axis("off")

# Slope Map
axs[1].imshow(grid_map.tensors["slopes"].cpu().numpy(), cmap="turbo")
axs[1].set_title("Slope Map")
axs[1].axis("off")

# Terrain Class Map
terrain_class_map = axs[2].imshow(
    grid_map.tensors["t_classes"].cpu().numpy(), cmap="jet"
)
axs[2].set_title("Terrain Class Map")
axs[2].axis("off")

# Color Map
color_map = axs[3].imshow(grid_map.tensors["colors"].cpu().numpy().transpose(1, 2, 0))
axs[3].set_title("Color Map")
axs[3].axis("off")

plt.show()
