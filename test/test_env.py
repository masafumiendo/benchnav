"""
author: Masafumi Endo
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from environments.env import GridMap
from environments.env import TerrainGeometry
from environments.env import TerrainColoring

# Initialize GridMap
grid_size = 64
resolution = 0.5
roughness_exponent = 0.8
amplitude_gain = 10
seed = 1

grid_map = GridMap(grid_size, resolution, roughness_exponent, amplitude_gain, seed)

# Set Terrain Geometry
terrain_geometry = TerrainGeometry(grid_map)
terrain_geometry.set_terrain_geometry(is_fractal=True,
                                      is_crater=True,
                                      num_craters=3,
                                      crater_margin=5,
                                      min_angle=20,
                                      max_angle=25,
                                      min_radius=5,
                                      max_radius=10
                                      )

# Set Terrain Coloring
terrain_coloring = TerrainColoring(grid_map)
occupancy = [0.4, 0.1, 0.4, 0.1]  # Example occupancy ratios for terrain types
threshold = 0.8
upper_threshold = 1
ambient_intensity = 0.1
terrain_coloring.set_terrain_class_distribution(occupancy, threshold, upper_threshold, ambient_intensity, seed)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Height Map
axs[0].imshow(grid_map.data.height, cmap='turbo')
axs[0].set_title('Height Map')
axs[0].axis('off')

# Terrain Class Map
terrain_class_map = axs[1].imshow(grid_map.data.t_class.reshape((grid_size, grid_size)), cmap='viridis')
axs[1].set_title('Terrain Class Map')
axs[1].axis('off')

# Color Map
color_map = axs[2].imshow(grid_map.data.color)
axs[2].set_title('Color Map')
axs[2].axis('off')

plt.show()

