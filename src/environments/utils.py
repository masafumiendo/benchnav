"""
author: Masafumi Endo
"""

from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import NDArray
import numpy as np


@dataclass
class Data:
    """
    Structure containing map-dependent data for terrain information, including height, slope, slip, terrain class, and color. 
    This class is designed to hold numpy arrays representing different aspects of the terrain being modeled.

    Attributes:
    - height (Optional[NDArray[np.float_]]): A 2D numpy array representing the terrain height map.
    - slope (Optional[NDArray[np.float_]]): A 2D numpy array representing the terrain slope values.
    - slip (Optional[NDArray[np.float_]]): A 2D numpy array representing slip ratios across the terrain.
    - t_class (Optional[NDArray[np.int_]]): A 2D numpy array representing terrain classification, where each class is indicated by one-hot encoding.
    - color (Optional[NDArray[np.float_]]): A 3D numpy array (height x width x 3) representing the RGB color values for visualizing the terrain.
    """

    height: Optional[NDArray[np.float_]] = field(default=None)
    slope: Optional[NDArray[np.float_]] = field(default=None)
    slip: Optional[NDArray[np.float_]] = field(default=None)
    t_class: Optional[NDArray[np.int_]] = field(default=None)
    color: Optional[NDArray[np.float_]] = field(default=None)
