"""
Kohei Honda, 2023.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np

import torch
import gymnasium as gym


class PlanetaryExplorationEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize PlanetaryExplorationEnv class.

        Parameters:
        - seed (Optional[int]): Random seed for reproducibility.
        - device (Optional[str]): Device to run the model on.
        """
        self._device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._seed = seed

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step the environment.

        Parameters:
        - action (np.ndarray): Action to take.

        Returns:
        - observation (np.ndarray): Observation after the action.
        - reward (float): Reward after the action.
        - terminated (bool): Whether the episode is terminated.
        - trucated (bool): Whether the episode is truncated.
        - info (dict): Additional information.

        """
        raise NotImplementedError

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        Parameters:
        - seed (Optional[int]): Random seed for reproducibility.
        - options (Optional[dict]): Additional options.

        Returns:
        - observation (np.ndarray): Initial observation.
        """
        raise NotImplementedError

    def render(self) -> None:
        """
        Render the environment.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the environment.
        """
        raise NotImplementedError
