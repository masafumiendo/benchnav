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
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor from the model as logits.
        """
        return super(Unet, self).forward(x)

    def predict(
        self, x: torch.Tensor, return_probabilities: bool = False
    ) -> torch.Tensor:
        """
        Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Parameters:
        - x (torch.Tensor): 4D torch tensor with shape (batch_size, channels, height, width)
        - return_probabilities (bool): If True, return the softmax probabilities instead of class indices.

        Returns:
        - torch.Tensor: The predicted segmentation map. Returns class indices by default, probabilities if return_probabilities is True.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            if return_probabilities:
                return probabilities
            else:
                return probabilities.argmax(dim=1)
