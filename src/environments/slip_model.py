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
        slope_noise_scale: float = 0,
        phi_bound: float = 45,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize SlipModelTorch class.

        Parameters:
        - device (Optional[str]): Device to run the model on.
        - slip_sensitivity (float): Sensitivity of slip to slope.
        - slip_nonlinearity (float): Nonlinearity of slip to slope.
        - slip_offset (float): Offset of slip.
        - base_noise_scale (float): Base scale of noise.
        - slope_noise_scale (float): Scale of noise increase with slope.
        - phi_bound (float): Bound of slope angle to generate heteroscedastic noise scales (-phi_bound, phi_bound).
        - seed (Optional[int]): Random seed for reproducibility.
        """
        self.device = (
            device
            if device is not None
            else "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.slip_sensitivity = slip_sensitivity
        self.slip_nonlinearity = slip_nonlinearity
        self.slip_offset = slip_offset
        self.base_noise_scale = base_noise_scale
        self.slope_noise_scale = slope_noise_scale
        self.phis_range = torch.linspace(-phi_bound, phi_bound, 100).to(self.device)
        self.heteroscedastic_noise_scales = (
            self.initialize_heteroscedastic_noise_scales()
        )
        set_randomness(seed) if seed is not None else None

    def initialize_heteroscedastic_noise_scales(self) -> Tensor:
        """
        Initialize the heteroscedastic noise scale for the model.
        
        Returns:
        - noise_scale (Tensor): Heteroscedastic noise scale for the model.
        """
        noise_scales = (
            self.base_noise_scale + torch.abs(self.phis_range) * self.slope_noise_scale
        )
        return noise_scales

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
        phi_tensor = (
            phi if isinstance(phi, Tensor) else torch.tensor([phi], dtype=torch.float32)
        )

        # Find the noise scale for the given slope angle
        indices = torch.argmin(torch.abs(self.phis_range[:, None] - phi_tensor), dim=0)
        noise_scales = self.heteroscedastic_noise_scales[indices]

        # Generate noise and add it to the slip
        noisy_slip = slip + torch.randn_like(slip) * noise_scales
        return (
            torch.clamp(noisy_slip, -1, 1).squeeze()
            if not isinstance(phi, Tensor)
            else torch.clamp(noisy_slip, -1, 1)
        )

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
