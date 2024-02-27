"""
author: Masafumi Endo
"""

from dataclasses import dataclass, field
from typing import Optional
from torch import Tensor


@dataclass
class TensorData:
    """
    Structure containing map-dependent data for terrain information, including height, slope, slip, terrain class, and color.
    This class is designed to hold PyTorch tensors representing different aspects of the terrain being modeled.

    Attributes:
    - heights (Optional[Tensor]): A 2D tensor representing the terrain height map.
    - slopes (Optional[Tensor]): A 2D tensor representing the terrain slope values.
    - slips (Optional[Tensor]): A 2D tensor representing slip ratios across the terrain.
    - t_classes (Optional[Tensor]): A 2D tensor representing terrain classification, where each class is indicated by one-hot encoding.
    - colors (Optional[Tensor]): A 3D tensor (height x width x 3) representing the RGB color values for visualizing the terrain.
    """

    heights: Optional[Tensor] = field(default=None)
    slopes: Optional[Tensor] = field(default=None)
    slips: Optional[Tensor] = field(default=None)
    t_classes: Optional[Tensor] = field(default=None)
    colors: Optional[Tensor] = field(default=None)
