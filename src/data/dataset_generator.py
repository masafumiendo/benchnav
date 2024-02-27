"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from typing import Tuple
import torch
import multiprocessing

from environments.grid_map import GridMap
from environments.grid_map import TerrainGeometry
from environments.grid_map import TerrainColoring
from src.data.utils import ParamsTerrainGeometry
from src.data.utils import ParamsTerrainColoring
from src.utils.utils import set_randomness


class DatasetGenerator:
    def __init__(
        self,
        data_directory: str,
        data_split: int,
        grid_size: int,
        resolution: float,
        environment_count: int,
        instance_count: int,
        num_total_terrain_classes: int = 10,
        num_selected_terrain_classes: int = 4,
        params_terrain_geometry: ParamsTerrainGeometry = None,
        params_terrain_coloring: ParamsTerrainColoring = None,
    ) -> None:
        """
        Initializes the DatasetGenerator with the specified parameters.

        Parameters:
        - data_directory (str): The directory to save the dataset.
        - data_split (int): The dataset split ('train', 'valid', 'test').
        - grid_size (int): The size of the grid map.
        - resolution (float): The resolution of the grid map.
        - environment_count (int): The number of environments to generate with the same terrain class distribution.
        - instance_count (int): The number of instances to generate for each environment.
        - num_total_terrain_classes (int): The total number of distinct terrain classes.
        - num_selected_terrain_classes (int): The number of terrain classes to select for each environment.
        - params_terrain_geometry (ParamsTerrainGeometry): The parameters for the terrain geometry.
        - params_terrain_coloring (ParamsTerrainColoring): The parameters for the terrain coloring.
        """
        # Validate the data_split argument
        if data_split not in ["train", "valid", "test"]:
            raise ValueError("data_split must be one of 'train', 'valid', 'test'")
        self.data_directory = data_directory
        self.data_split = data_split

        self.grid_size = grid_size
        self.resolution = resolution
        self.environment_count = environment_count
        self.instance_count = instance_count
        self.num_total_terrain_classes = num_total_terrain_classes
        self.num_selected_terrain_classes = num_selected_terrain_classes
        self.params_terrain_geometry = params_terrain_geometry
        self.params_terrain_coloring = params_terrain_coloring

    def generate_dataset(self, processes: int = 4) -> None:
        """
        Generates and saves a dataset with seed values for reproducibility.

        Parameters:
        - processes (int): The number of processes to use for parallel generation.
        """
        # Set seed for reproducibility
        environment_seed, base_instance_seed = self.set_seed()

        occupancies = self.generate_occupancy_distribution(seed=environment_seed)

        # Generate and save environment groups
        pbar = tqdm(
            total=self.environment_count,
            desc=f"Generating {self.data_split} environments",
        )
        with multiprocessing.Pool(processes=processes) as pool:
            for environment_index in range(self.environment_count):
                # Calculate global instance seed
                instance_seed = (
                    base_instance_seed + environment_index * self.instance_count
                )
                # Update occupancy for each environment group
                self.params_terrain_coloring.occupancy = occupancies[
                    environment_index, :
                ]
                pool.apply_async(
                    self.generate_and_save_environment_group,
                    args=(environment_index, instance_seed),
                    callback=lambda _: pbar.update(1),
                )
            # Close pool and wait for all processes to finish
            pool.close()
            pool.join()

        pbar.close()

        # Save environment and instance seeds in .npy file
        last_instance_seed = (
            base_instance_seed + self.environment_count * self.instance_count
        )
        file_path = os.path.join(
            self.data_directory, self.data_split, "seed_information.pt"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(
            {
                "environment_seed": environment_seed,
                "last_instance_seed": last_instance_seed,
            },
            file_path,
        )

    def set_seed(self) -> Tuple[int, int]:
        """
        Sets the seed for reproducibility.

        Returns:
        - Tuple[int, int]: The environment seed and the base instance seed.
        """
        if self.data_split == "train":
            environment_seed = 0
            base_instance_seed = 0
        elif self.data_split == "valid":
            seed_information = torch.load(
                os.path.join(self.data_directory, "train", "seed_information.pt")
            )
            environment_seed = seed_information["environment_seed"]
            base_instance_seed = seed_information["last_instance_seed"]
        elif self.data_split == "test":
            seed_information = torch.load(
                os.path.join(self.data_directory, "valid", "seed_information.pt")
            )
            environment_seed = seed_information["environment_seed"]
            base_instance_seed = seed_information["last_instance_seed"]
        return environment_seed, base_instance_seed

    def generate_occupancy_distribution(self, seed: int) -> torch.Tensor:
        """
        Generates a distribution of terrain class occupancies for each environment using PyTorch.

        Parameters:
        - seed (int): The random seed for generating the occupancy distribution.

        Returns:
        - torch.Tensor: Occupancy distributions for every environment.
        """
        # Set seed for reproducibility
        set_randomness(seed)

        occupancies = torch.zeros(
            self.environment_count, self.num_total_terrain_classes
        )
        for environment_index in range(self.environment_count):
            selected_terrain_indices = torch.randperm(self.num_total_terrain_classes)[
                : self.num_selected_terrain_classes
            ]
            occupancies[environment_index, selected_terrain_indices] = (
                1 / self.num_selected_terrain_classes
            )

        self.balance_occupancy_distribution(occupancies)
        return occupancies

    def balance_occupancy_distribution(self, occupancies: torch.Tensor) -> None:
        """
        Balances the occupancy distribution to ensure that each terrain class has almost equal distribution using PyTorch.

        Parameters:
        - occupancies (torch.Tensor): The occupancy distributions for all environments.
        """
        total_selections = self.num_selected_terrain_classes * self.environment_count
        expected_per_class = total_selections / self.num_total_terrain_classes

        current_selections = torch.sum(
            occupancies == 1 / self.num_selected_terrain_classes, dim=0
        )

        # Balance occupancy distribution by swapping instances
        for _ in range(1000):
            overrepresented_classes = torch.where(
                current_selections > expected_per_class
            )[0]
            underrepresented_classes = torch.where(
                current_selections < expected_per_class
            )[0]

            if len(overrepresented_classes) == 0 or len(underrepresented_classes) == 0:
                break

            for over in overrepresented_classes:
                for under in underrepresented_classes:
                    swap_candidates = torch.nonzero(
                        (occupancies[:, over] == 1 / self.num_selected_terrain_classes)
                        & (occupancies[:, under] == 0),
                        as_tuple=False,
                    ).squeeze()

                    if swap_candidates.numel() > 0:
                        environment_index = swap_candidates[0].item()
                        occupancies[environment_index, over] = 0
                        occupancies[environment_index, under] = (
                            1 / self.num_selected_terrain_classes
                        )
                        current_selections[over] -= 1
                        current_selections[under] += 1
                        break

    def generate_and_save_environment_group(
        self, environment_index: int, instance_seed: int
    ) -> None:
        """
        Generates and saves a group of environment instances with the specified parameters.

        Parameters:
        - environment_index (int): The index of the environment.
        - instance_seed (int): The seed for generating the map instance.
        """
        # Generate and save environment instances
        for instance_index in range(self.instance_count):
            # Generate map instance
            grid_map = self.generate_map_instance(seed=instance_seed)

            # Save map instance
            file_path = os.path.join(
                self.data_directory,
                self.data_split,
                f"{environment_index:03d}_{instance_index:03d}.pt",
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            torch.save(grid_map.tensor_data, file_path)

            # Update instance seed
            instance_seed += 1

    def generate_map_instance(self, seed: int) -> GridMap:
        """
        Generates a map instance with the specified geometry and coloring parameters.

        Parameters:
        - seed (int): The random seed for generating the map instance.

        Returns:
        - GridMap: The generated map instance.
        """

        # Initialize GridMap
        grid_map = GridMap(
            grid_size=self.grid_size, resolution=self.resolution, seed=seed
        )

        # Set Terrain Geometry
        params_terrain_geometry = self.params_terrain_geometry
        terrain_geometry = TerrainGeometry(grid_map)
        terrain_geometry.set_terrain_geometry(
            params_terrain_geometry.is_fractal,
            params_terrain_geometry.is_crater,
            params_terrain_geometry.num_craters,
            params_terrain_geometry.crater_margin,
            params_terrain_geometry.min_angle,
            params_terrain_geometry.max_angle,
            params_terrain_geometry.min_radius,
            params_terrain_geometry.max_radius,
        )

        # Set Terrain Coloring
        params_terrain_coloring = self.params_terrain_coloring
        terrain_coloring = TerrainColoring(grid_map)
        terrain_coloring.set_terrain_class_coloring(
            params_terrain_coloring.occupancy,
            params_terrain_coloring.lower_threshold,
            params_terrain_coloring.upper_threshold,
            params_terrain_coloring.ambient_intensity,
        )

        return grid_map
