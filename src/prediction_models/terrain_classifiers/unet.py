"""
author: Masafumi Endo
"""

import torch
import segmentation_models_pytorch as smp


class Unet(smp.Unet):
    def __init__(
        self,
        classes: int,
        encoder_name: str = "resnet18",
        encoder_weights: str = "imagenet",
    ) -> None:
        """
        Initialize UNet class.

        Parameters:
        - encoder_name (str): Name of the encoder.
        - encoder_weights (str): Weights of the encoder.
        - classes (int): Number of classes for segmentation.
        """
        super(Unet, self).__init__(
            encoder_name=encoder_name, encoder_weights=encoder_weights, classes=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UNet model.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor from the model as logits.
        """
        return super(Unet, self).forward(x)
