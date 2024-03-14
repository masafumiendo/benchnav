"""
Masafumi Endo, 2024
"""

import torch
from torch import Tensor
from torch.distributions import Normal
from typing import Optional, Union

from src.utils.utils import set_randomness


class SlipModel:
    def __init__(
        self,
        slip_sensitivity: float,
        slip_nonlinearity: float,
        slip_offset: float,
        base_noise_scale: float = 0.05,
        slope_noise_scale: float = 0.00,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize SlipModel class.

        Parameters:
        - slip_sensitivity (float): Sensitivity of slip to slope.
        - slip_nonlinearity (float): Nonlinearity of slip to slope.
        - slip_offset (float): Offset of slip.
        - noise_scale (float): Scale of noise.
        - seed (Optional[int]): Random seed for reproducibility.
        - device (Optional[str]): Device to run the model on.
        """
        self.slip_sensitivity = slip_sensitivity
        self.slip_nonlinearity = slip_nonlinearity
        self.slip_offset = slip_offset
        self.base_noise_scale = base_noise_scale
        self.slope_noise_scale = slope_noise_scale
        set_randomness(seed) if seed is not None else None
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        # Initialize the distribution
        self.distribution = None

    def sample(self, sample_shape: Optional[int] = 1) -> Tensor:
        """
        Sample slip values from the model distribution.

        Parameters:
        - sample_shape (Optional[int]): Number of samples to draw (default: 1).

        Returns:
        - slip (Tensor): Sampled slip values.
        """
        if self.distribution is None:
            raise ValueError("Model distribution is not defined.")
        samples = self.distribution.sample(sample_shape=torch.Size([sample_shape]))
        return torch.clamp(samples, 0, 1)

    def model_distribution(self, phi: Tensor) -> Normal:
        """
        Define the distribution of slip from slope angle, without noise.
        This method supports only tensor inputs for the slope angle.

        Parameters:
        - phi (Tensor): Slope angle, can be a tensor of values.

        Returns:
        - dist (Normal): Slip distribution as a normal distribution.
        """
        phi_tensor = self.ensure_tensor(phi)
        mean = self.model_mean(phi_tensor)
        stddev = self.model_stddev(phi_tensor)
        self.distribution = Normal(mean, stddev)
        return self.distribution

    def model_mean(self, phi: Tensor) -> Tensor:
        """
        Define the predictive mean of the latent slip model from slope angle, without noise.
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
        return torch.clamp(slip, 0, 1)

    def model_stddev(self, phi: Tensor) -> Tensor:
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
