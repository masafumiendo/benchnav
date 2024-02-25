"""
author: Masafumi Endo
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class ParamsTerrainGeometry:
    """
    A structure containing parameters for terrain geometry generation, specifying how terrain features like craters and fractal geometries are generated.

    Attributes:
    - is_fractal (Optional[bool]): Specifies whether fractal geometry is used in terrain generation.
    - is_crater (Optional[bool]): Specifies whether craters are included in the terrain.
    - num_craters (Optional[int]): The number of craters to generate. Must be greater than 0 if craters are enabled.
    - crater_margin (Optional[int]): The margin around craters, affecting their blending with the terrain.
    - min_angle (Optional[int]): The minimum angle for terrain features, ensuring diverse geometric shapes.
    - max_angle (Optional[int]): The maximum angle for terrain features. Must be greater than `min_angle`.
    - min_radius (Optional[int]): The minimum radius for circular terrain features, such as craters.
    - max_radius (Optional[int]): The maximum radius for circular terrain features. Must be greater than `min_radius`.
    """

    is_fractal: Optional[bool] = field(default=None)
    is_crater: Optional[bool] = field(default=None)
    num_craters: Optional[int] = field(default=None)
    crater_margin: Optional[int] = field(default=None)
    min_angle: Optional[int] = field(default=None)
    max_angle: Optional[int] = field(default=None)
    min_radius: Optional[int] = field(default=None)
    max_radius: Optional[int] = field(default=None)

    def __post_init__(self):
        # Validate crater parameters
        if self.is_crater and (self.num_craters is None or self.num_craters <= 0):
            raise ValueError("num_craters must be greater than 0 if is_crater is True.")

        # Validate min/max angles
        if (
            self.min_angle is not None
            and self.max_angle is not None
            and self.min_angle >= self.max_angle
        ):
            raise ValueError("max_angle must be greater than min_angle.")

        # Validate min/max radii
        if (
            self.min_radius is not None
            and self.max_radius is not None
            and self.min_radius >= self.max_radius
        ):
            raise ValueError("max_radius must be greater than min_radius.")


@dataclass
class ParamsTerrainColoring:
    """
    A structure containing parameters for terrain coloring generation. This class specifies how colors are applied to terrain based on various parameters such as occupancy, color thresholds, and ambient light intensity.

    Attributes:
    - occupancy (Optional[NDArray[np.float_]]): A occupancy percentages array as floats that influence terrain color distribution. The sum of these values must exactly equal 1, representing a complete distribution across defined occupancy levels.
    - lower_threshold (Optional[float]): The minimum threshold value for color application, below which colors are not applied. Must be a value between 0 and 1.
    - upper_threshold (Optional[float]): The maximum threshold value for color application, above which colors are not applied. Must be greater than the lower_threshold and within the range of 0 to 1.
    - ambient_intensity (Optional[float]): The intensity of ambient light affecting the terrain color. This parameter is flexible and can be any float value, with no specific range constraints.
    """

    occupancy: Optional[NDArray[np.float_]] = field(default=None)
    lower_threshold: Optional[float] = field(default=None)
    upper_threshold: Optional[float] = field(default=None)
    ambient_intensity: Optional[float] = field(default=None)

    def __post_init__(self):
        # Validate total occupancy equals 1 strictly
        if self.occupancy and self.occupancy.sum() != 1:
            raise ValueError("The sum of occupancy values must exactly equal 1.")

        # Validate lower and upper thresholds
        if self.lower_threshold is not None and not (0 <= self.lower_threshold <= 1):
            raise ValueError("Lower threshold must be between 0 and 1.")
        if self.upper_threshold is not None and not (0 <= self.upper_threshold <= 1):
            raise ValueError("Upper threshold must be between 0 and 1.")
        if (self.lower_threshold is not None and self.upper_threshold is not None) and (
            self.lower_threshold >= self.upper_threshold
        ):
            raise ValueError("Upper threshold must be greater than lower threshold.")
