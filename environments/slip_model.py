"""
author: Masafumi Endo
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union

class SlipModel:
    def __init__(
        self,
        slip_sensitivity: float,
        slip_nonlinearity: float,
        slip_offset: float,
        noise_scale: float = 0.05,
        seed: Optional[int] = None) -> None:
        """
        Initialize SlipModel class.

        Parameters:
        
        - slip_sensitivity (float): Sensitivity of slip to slope.
        - slip_nonlinearity (float): Nonlinearity of slip to slope.
        - slip_offset (float): Offset of slip.
        - noise_scale (float): Scale of noise.
        - seed (Optional[int]): Random seed for reproducibility.
        """
        self.slip_sensitivity = slip_sensitivity
        self.slip_nonlinearity = slip_nonlinearity
        self.slip_offset = slip_offset
        self.noise_scale = noise_scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def observe_slip(self, phi: Union[float, NDArray[np.float_]]) -> Union[float, NDArray[np.float_]]:
        """
        Observe slip from slope angle, ensuring the result is within the range (-1, 1).
        This method supports both float and array inputs for the slope angle.

        Parameters:
        - phi (Union[float, NDArray[np.float_]]): Slope angle, can be a single value or an array of values.

        Returns:
        - slip (Union[float, NDArray[np.float_]]): Slip ratio, adjusted to be within the range (-1, 1),
                                               can be a single value or an array of values depending on the input.
        """
        slip = self.latent_model(phi)
        # Generate slip with noise and then clip it to be within the range (-1, 1)
        noisy_slip = slip + self.rng.normal(0, self.noise_scale, size=np.shape(phi))
        return np.clip(noisy_slip, -1, 1)
    
    def latent_model(self, phi: Union[float, NDArray[np.float_]]) -> Union[float, NDArray[np.float_]]:
        """
        Define the latent model of slip from slope angle, without noise.
        This method supports both float and array inputs for the slope angle.

        Parameters:
        - phi (Union[float, NDArray[np.float_]]): Slope angle, can be a single value or an array of values.

        Returns:
        - slip (Union[float, NDArray[np.float_]]): Slip ratio, without noise,
                                               can be a single value or an array of values depending on the input.
        """
        base_slip = self.slip_sensitivity * 1e-3 * np.abs(phi)**self.slip_nonlinearity
        # Apply conditional operation based on the sign of phi, supporting both scalar and array inputs
        slip = np.where(phi >= 0, base_slip + self.slip_offset, -base_slip + self.slip_offset)
        return slip