"""
author: Masafumi Endo
"""

import os
from typing import Tuple
import torch
from torch.utils.data import Dataset


class TerrainDataset(Dataset):
    """
    A dataset class for loading terrain data for machine learning models.
    Supports loading color maps and mask maps as well as converting them into a format suitable for model training or evaluation.
    """

    def __init__(self, data_directory: str, data_split: str):
        """
        Initializes the TerrainDataset with the specified directory, and data split.

        Parameters:
        - data_directory (str): The directory containing the dataset.
        - data_split (str): The dataset split ('train', 'valid', 'test').
        """
        # Validate the data_split argument
        if data_split not in ["train", "valid", "test"]:
            raise ValueError("data_split must be one of 'train', 'valid', 'test'")

        self.data_directory = os.path.join(data_directory, data_split + "/")
        self.data_indices = [
            file
            for file in os.listdir(self.data_directory)
            if file != "seed_information.npy"
        ]
        self.file_paths = [
            os.path.join(self.data_directory, file_id) for file_id in self.data_indices
        ]

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the color map and mask map for the specified index.

        Parameters:
        - index (int): The index of the data item.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: containing the color map and the mask map.
        """
        data_item = torch.load(self.file_paths[index])
        color_map = data_item["colors"]
        terrain_class = data_item["t_classes"]
        return color_map, terrain_class
