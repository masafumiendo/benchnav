"""
author: Masafumi Endo
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Data:
    """
    Structure containing map-dependent data.
    """

    height: Optional[np.ndarray] = field(default=None)
    slope: Optional[np.ndarray] = field(default=None)
    slip: Optional[np.ndarray] = field(default=None)
    t_class: Optional[np.ndarray] = field(default=None)
    color: Optional[np.ndarray] = field(default=None)
