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
        device: Optional[str],
        slip_sensitivity: float,
        slip_nonlinearity: float,
        slip_offset: float,
        base_noise_scale: float = 0.05,
        slope_noise_scale: float = 0.00,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize SlipModelTorch class.

        Parameters:
        - device (Optional[str]): Device to run the model on.
        - slip_sensitivity (float): Sensitivity of slip to slope.
        - slip_nonlinearity (float): Nonlinearity of slip to slope.
        - slip_offset (float): Offset of slip.
        - noise_scale (float): Scale of noise.
        - seed (Optional[int]): Random seed for reproducibility.
        """
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.slip_sensitivity = slip_sensitivity
        self.slip_nonlinearity = slip_nonlinearity
        self.slip_offset = slip_offset
        self.base_noise_scale = base_noise_scale
        self.slope_noise_scale = slope_noise_scale
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
        phi_tensor = self.ensure_tensor(phi)
        slip = self.latent_model(phi_tensor)
        noise_scales = self.noise_model(phi_tensor)
        # Generate slip with noise and then clip it to be within the range (-1, 1)
        noisy_slip = torch.clamp(slip + torch.randn_like(slip) * noise_scales, -1, 1)
        return noisy_slip if isinstance(phi, Tensor) else noisy_slip.item()

    def latent_model(self, phi: Tensor) -> Tensor:
        """
        Define the latent model of slip from slope angle, without noise.
        This method supports only tensor inputs for the slope angle.

        Parameters:
        - phi (Tensor): Slope angle, can be a tensor of values.

        Returns:
        - slip (Tensor): Slip ratio as a tensor.
        """
        base_slip = (
            self.slip_sensitivity * 1e-3 * torch.abs(phi) ** self.slip_nonlinearity
        )
        # Apply conditional operation based on the sign of phi, supporting both scalar and tensor inputs
        slip = torch.where(
            phi >= 0, base_slip + self.slip_offset, -base_slip + self.slip_offset
        )
        return torch.clamp(slip, -1, 1)

    def noise_model(self, phi: Tensor) -> Tensor:
        """
        Define the noise model of slip from slope angle, without noise.
        This method supports only tensor inputs for the slope angle.

        Parameters:
        - phi (Tensor): Slope angle, can be a tensor of values.

        Returns:
        - noise (Tensor): Heteroscedastic noise scale as a tensor.
        """
        noise_scales = self.base_noise_scale + self.slope_noise_scale * torch.abs(phi)
        return noise_scales

    def ensure_tensor(self, x: Union[float, Tensor]) -> Tensor:
        """
        Ensure the input is a tensor.

        Parameters:
        - x (Union[float, Tensor]): Input, can be a single value or a tensor of values.

        Returns:
        - x (Tensor): Input, as a tensor.
        """
        if isinstance(x, float) or isinstance(x, int):
            return torch.tensor(x, device=self.device, dtype=torch.float32)
        return x.to(self.device)
