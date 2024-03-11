"""
author: Masafumi Endo
"""

import torch
import gpytorch

from typing import Tuple


class GPModel(gpytorch.models.ExactGP):
    """
    This class defines an exact Gaussian Process Model using GPyTorch.
    It utilizes a constant mean function and an RBF kernel with automatic relevance determination (ARD)
    for the covariance function, suitable for a wide range of regression tasks.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ) -> None:
        """
        Initializes the GPModel with training data, target values, and a likelihood model.

        Parameters:
        - train_x (Tensor): The training input data.
        - train_y (Tensor): The training target data.
        - likelihood (gpytorch.likelihoods.Likelihood): The likelihood model to use for inference.
        """
        super(GPModel, self).__init__(train_x, train_y, likelihood)

        # Define the mean function as a constant mean.
        self.mean_module = gpytorch.means.ConstantMean()

        # Define the covariance function (kernel) as an RBF kernel with ARD.
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        The forward pass for generating predictions.

        Parameters:
        - x (Tensor): The input data for which predictions are to be generated.

        Returns:
        - MultivariateNormal: The predictive distribution of the model.
        """
        # Compute the mean of the given inputs.
        mean_x = self.mean_module(x)
        # Compute the covariance of the given inputs.
        covar_x = self.covar_module(x)
        # Return the distribution of predicted values.
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Generates predictions for the given input data.

        Parameters:
        - x (Tensor): The input data for which predictions are to be generated.

        Returns:
        - MultivariateNormal: The predictive distribution of the model.
        """
        # Set the model to evaluation mode
        self.eval()
        # Set the likelihood to evaluation mode
        self.likelihood.eval()
        # Generate predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self(x))
        return pred_dist
