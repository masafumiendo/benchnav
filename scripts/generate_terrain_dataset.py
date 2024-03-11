"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.multiprocessing as multiprocessing

from src.data.dataset_generator import DatasetGenerator
from src.data.utils import ParamsTerrainGeometry
from src.data.utils import ParamsTerrainColoring
from src.data.slip_models_generator import SlipModelsGenerator


def main(device: str):
    # Initialize the shared parameters
    dataset_index = 1
    subset_index = 1
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(
        script_directory, f"datasets/dataset{dataset_index:02d}/"
    )
    grid_size = 64
    resolution = 0.5
    environment_count = 10

    # Set the parameters for the slip models
    slip_sensitivity_minmax = (1.0, 9.0)
    slip_nonlinearity_minmax = (1.4, 2.0)
    slip_offset_minmax = (0.0, 0.1)
    noise_scale_minmax = (0.1, 0.2)

    # Generate the slip models
    slip_models_generator = SlipModelsGenerator(
        num_total_terrain_classes=10,
        slip_sensitivity_minmax=slip_sensitivity_minmax,
        slip_nonlinearity_minmax=slip_nonlinearity_minmax,
        slip_offset_minmax=slip_offset_minmax,
        noise_scale_minmax=noise_scale_minmax,
        device=device,
    )
    slip_models = slip_models_generator.generate_slip_models()

    # Set the parameters for the terrain geometry
    params_terrain_geometry = ParamsTerrainGeometry(is_fractal=True, is_crater=False)

    # Set the parameters for the terrain coloring
    params_terrain_coloring = ParamsTerrainColoring(
        lower_threshold=0.8, upper_threshold=1, ambient_intensity=0.1
    )

    # Generate the dataset for the training split
    generator_train = DatasetGenerator(
        slip_models=slip_models,
        data_directory=data_directory,
        data_split="train",
        grid_size=grid_size,
        resolution=resolution,
        environment_count=environment_count,
        instance_count=100,
        params_terrain_geometry=params_terrain_geometry,
        params_terrain_coloring=params_terrain_coloring,
        device=device,
    )
    generator_train.generate_dataset()

    # Generate the dataset for the validation split
    generator_valid = DatasetGenerator(
        slip_models=slip_models,
        data_directory=data_directory,
        data_split="valid",
        grid_size=grid_size,
        resolution=resolution,
        environment_count=environment_count,
        instance_count=25,
        params_terrain_geometry=params_terrain_geometry,
        params_terrain_coloring=params_terrain_coloring,
        device=device,
    )
    generator_valid.generate_dataset()

    # Generate the dataset for the test split
    generator_test = DatasetGenerator(
        slip_models=slip_models,
        data_directory=data_directory,
        data_split="test",
        grid_size=grid_size,
        resolution=resolution,
        environment_count=environment_count,
        instance_count=10,
        params_terrain_geometry=params_terrain_geometry,
        params_terrain_coloring=params_terrain_coloring,
        subset_index=subset_index,
        device=device,
    )
    generator_test.generate_dataset()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(device="cuda")
