"""
author: Masafumi Endo
"""

import torch
from torch import Tensor
from typing import Optional, Union

from src.utils.utils import set_randomness


class SlipModel:
    def __init__(
        self,
        slip_sensitivity: float,
        slip_nonlinearity: float,
        slip_offset: float,
        noise_scale: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize SlipModelTorch class.

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
        set_randomness(seed) if seed is not None else None

    def observe_slip(self, phi: Union[float, Tensor]) -> Union[float, Tensor]:
        """
        Observe slip from slope angle, ensuring the result is within the range (-1, 1).
        This method supports both float and tensor inputs for the slope angle.

        Parameters:
        - phi (Union[float, Tensor]): Slope angle, can be a single value or a tensor of values.

        Returns:
        - slip (Union[float, Tensor]): Slip ratio, adjusted to be within the range (-1, 1),
                                       can be a single value or a tensor of values depending on the input.
        """
        slip = self.latent_model(phi)
        # Generate slip with noise and then clip it to be within the range (-1, 1)
        if isinstance(phi, Tensor):
            noisy_slip = slip + torch.normal(
                0, self.noise_scale, size=phi.shape, device=phi.device
            )
        else:
            noisy_slip = slip + torch.normal(0, self.noise_scale, size=(1,))
        return torch.clamp(noisy_slip, -1, 1)

    def latent_model(self, phi: Union[float, Tensor]) -> Union[float, Tensor]:
        """
        Define the latent model of slip from slope angle, without noise.
        This method supports both float and tensor inputs for the slope angle.

        Parameters:
        - phi (Union[float, Tensor]): Slope angle, can be a single value or a tensor of values.

        Returns:
        - slip (Union[float, Tensor]): Slip ratio, without noise,
                                       can be a single value or a tensor of values depending on the input.
        """
        base_slip = (
            self.slip_sensitivity * 1e-3 * torch.abs(phi) ** self.slip_nonlinearity
        )
        # Apply conditional operation based on the sign of phi, supporting both scalar and tensor inputs
        slip = torch.where(
            phi >= 0, base_slip + self.slip_offset, -base_slip + self.slip_offset
        )
        return torch.clamp(slip, -1, 1)
