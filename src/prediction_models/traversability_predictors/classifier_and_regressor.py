"""
Masafumi Endo, 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional
import torch
import gpytorch
from torch.distributions import Normal


class TraversabilityPredictor:
    def __init__(
        self,
        terrain_classifier: torch.nn.Module,
        slip_regressors: Dict[int, gpytorch.Module],
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the TraversabilityPredictor class.

        Parameters:
        - terrain_classifier (torch.nn.Module): the terrain classifier.
        - slip_regressors (Dict[int, gpytorch.Module]): the slip regressors for each terrain class.
        - device (Optional[str]): the device to use for inference.
        """
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.terrain_classifier = terrain_classifier.to(self.device)
        self.slip_regressors = {
            terrain_class: regressor.to(self.device)
            for terrain_class, regressor in slip_regressors.items()
        }

    def predict(self, colors, slopes) -> Normal:
        """
        Predict the traversability of the terrain.

        Parameters:
        - colors (torch.Tensor): the color maps of the terrain.
        - slopes (torch.Tensor): the slope maps of the terrain.

        Returns:
        - Normal: the predictive distribution of the traversability.
        """
        # Move the data to the device
        colors = colors.to(self.device)
        slopes = slopes.to(self.device)

        # Predict the terrain class
        t_classes = self.terrain_classifier.predict(colors)

        # Predict the slip for each terrain class
        slip_mean = torch.zeros_like(t_classes, dtype=torch.float32)
        slip_stddev = torch.zeros_like(t_classes, dtype=torch.float32)

        for t_class, slip_regressor in self.slip_regressors.items():
            mask = t_classes == t_class
            if mask.any():
                slip_dist = slip_regressor.predict(slopes[mask])
                slip_mean[mask] = slip_dist.mean
                slip_stddev[mask] = slip_dist.stddev

        # Return the predictive distribution of the traversability
        return Normal(slip_mean, slip_stddev)
