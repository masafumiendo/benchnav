"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_generator import DatasetGenerator
from src.data.utils import ParamsTerrainGeometry
from src.data.utils import ParamsTerrainColoring


def main():
    # Initialize the shared parameters
    dataset_index = 1
    subset_index = 1
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(
        script_directory,
        f"datasets/dataset{dataset_index:02d}/subset{subset_index:02d}/",
    )
    grid_size = 64
    resolution = 0.5
    environment_count = 10

    # Set the parameters for the terrain geometry
    params_terrain_geometry = ParamsTerrainGeometry(is_fractal=True, is_crater=False)

    # Set the parameters for the terrain coloring
    params_terrain_coloring = ParamsTerrainColoring(
        lower_threshold=0.8, upper_threshold=1, ambient_intensity=0.1
    )

    # Generate the dataset for the training split
    generator_train = DatasetGenerator(
        data_directory=data_directory,
        data_split="train",
        grid_size=grid_size,
        resolution=resolution,
        environment_count=environment_count,
        instance_count=100,
        params_terrain_geometry=params_terrain_geometry,
        params_terrain_coloring=params_terrain_coloring,
    )
    generator_train.generate_dataset()

    # Generate the dataset for the validation split
    generator_valid = DatasetGenerator(
        data_directory=data_directory,
        data_split="valid",
        grid_size=grid_size,
        resolution=resolution,
        environment_count=environment_count,
        instance_count=25,
        params_terrain_geometry=params_terrain_geometry,
        params_terrain_coloring=params_terrain_coloring,
    )
    generator_valid.generate_dataset()

    # Generate the dataset for the test split
    generator_test = DatasetGenerator(
        data_directory=data_directory,
        data_split="test",
        grid_size=grid_size,
        resolution=resolution,
        environment_count=environment_count,
        instance_count=10,
        params_terrain_geometry=params_terrain_geometry,
        params_terrain_coloring=params_terrain_coloring,
    )
    generator_test.generate_dataset()


if __name__ == "__main__":
    main()
