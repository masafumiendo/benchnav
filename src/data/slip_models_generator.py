"""
Masafumi Endo, 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Tuple, Union, Optional, Dict
from collections import defaultdict

from src.environments.slip_model import SlipModel
from src.utils.utils import set_randomness


class SlipModelsGenerator:
    def __init__(
        self,
        num_total_terrain_classes: int,
        slip_sensitivity_minmax: Tuple[float, float],
        slip_nonlinearity_minmax: Tuple[float, float],
        slip_offset_minmax: Tuple[float, float],
        noise_scale_minmax: Tuple[float, float],
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the SlipModelsGenerator class.

        Parameters:
        - num_total_terrain_classes (int): the number of terrain classes.
        - slip_sensitivity_minmax (Tuple[float, float]): the min and max value of the slip sensitivity.
        - slip_nonlinearity_minmax (Tuple[float, float]): the min and max value of the slip nonlinearity.
        - slip_offset_minmax (Tuple[float, float]): the min and max value of the slip offset.
        - noise_scale_minmax (Tuple[float, float]): the min and max value of the noise scale.
        - device (Optional[str]): the device to use for training.
        """
        self.num_total_terrain_classes = num_total_terrain_classes
        self.slip_sensitivity_minmax = slip_sensitivity_minmax
        self.slip_nonlinearity_minmax = slip_nonlinearity_minmax
        self.slip_offset_minmax = slip_offset_minmax
        self.noise_scale_minmax = noise_scale_minmax

        # check if the minmax values are valid
        self.validate_minmax(slip_sensitivity_minmax)
        self.validate_minmax(slip_nonlinearity_minmax)
        self.validate_minmax(slip_offset_minmax)
        self.validate_minmax(noise_scale_minmax)

        self.device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    def validate_minmax(self, minmax: Tuple[float, float]) -> None:
        """
        Validate the minmax values.

        Parameters:
        - minmax (Tuple[float, float]): the min and max values.
        """
        if minmax[0] >= minmax[1]:
            raise ValueError("The minimum value must be less than the maximum value.")

    def generate_slip_models(self) -> Dict[int, SlipModel]:
        """
        Generates slip models for each terrain class.

        Returns:
        - slip_models (Dict[int, SlipModel]): the slip models for each terrain class.
        """
        slip_models = defaultdict(SlipModel)
        for terrain_class in range(self.num_total_terrain_classes):
            slip_sensitivity, slip_nonlinearity, slip_offset, noise_scale = (
                self.set_slip_model_parameters(terrain_class)
            )
            slip_models[terrain_class] = SlipModel(
                slip_sensitivity=slip_sensitivity,
                slip_nonlinearity=slip_nonlinearity,
                slip_offset=slip_offset,
                base_noise_scale=noise_scale,
                seed=terrain_class,
                device=self.device,
            )

        return slip_models

    def set_slip_model_parameters(
        self, terrain_class: int
    ) -> Tuple[float, float, float, float]:
        """
        Set the slip model parameters for the given terrain class.

        Parameters:
        - terrain_class (int): the terrain class used as a seed for the random number generator.

        Returns:
        - slip_sensitivity (float): the slip sensitivity for the given terrain class
        - slip_nonlinearity (float): the slip nonlinearity for the given terrain class
        - slip_offset (float): the slip offset for the given terrain class
        - noise_scale (float): the noise scale for the given terrain class
        """
        set_randomness(terrain_class)
        slip_sensitivity = self.uniform_sampling(
            self.slip_sensitivity_minmax[0], self.slip_sensitivity_minmax[1]
        )
        slip_nonlinearity = self.uniform_sampling(
            self.slip_nonlinearity_minmax[0], self.slip_nonlinearity_minmax[1]
        )
        slip_offset = self.uniform_sampling(
            self.slip_offset_minmax[0], self.slip_offset_minmax[1]
        )
        noise_scale = self.uniform_sampling(
            self.noise_scale_minmax[0], self.noise_scale_minmax[1]
        )
        return slip_sensitivity, slip_nonlinearity, slip_offset, noise_scale

    def uniform_sampling(
        self, min_val: float, max_val: float, num_samples: int = 1
    ) -> Union[float, torch.Tensor]:
        """
        Uniformly sample the given range.

        Parameters:
        - min_val (float): the minimum value of the range
        - max_val (float): the maximum value of the range
        - num_samples (int): the number of samples

        Returns:
        - samples (Union[float, torch.Tensor]): the uniformly sampled values
        """
        samples = torch.rand(num_samples) * (max_val - min_val) + min_val
        return samples if num_samples > 1 else samples.item()
