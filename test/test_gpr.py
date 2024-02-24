"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from src.environments.slip_model import SlipModel
from src.prediction_models.slip_regressors.gpr import GPModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate the SlipModel
slip_model = SlipModel(slip_sensitivity=1.0, slip_nonlinearity=2.0, slip_offset=0.1)

# Generate synthetic data
phis = np.linspace(-30, 30, 1000)
observed_slips = slip_model.observe_slip(phis)

# Convert the data to PyTorch tensors
phis_tensor = torch.from_numpy(phis).to(device=DEVICE, dtype=torch.float32)
observed_slips_tensor = torch.from_numpy(observed_slips).to(
    device=DEVICE, dtype=torch.float32
)

# Instantiate the GPModel
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=DEVICE)
model = GPModel(phis_tensor, observed_slips_tensor, likelihood).to(device=DEVICE)

# Set the model and likelihood to training mode
model.train()
likelihood.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training loop
training_iterations = 50
for i in range(training_iterations):
    # Zero backprop gradients
    optimizer.zero_grad()
    # Get output from model
    output = model(phis_tensor)
    # Calc loss and backprop derivatives
    loss = -mll(output, observed_slips_tensor)
    loss.backward()
    print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item()}")
    optimizer.step()

# Set the model and likelihood to evaluation mode
model.eval()
likelihood.eval()

# Generate test data
test_phis = np.linspace(-30, 30, 100)
test_slips = slip_model.observe_slip(test_phis)

# Convert the test data to PyTorch tensors
test_phis_tensor = torch.from_numpy(test_phis).to(device=DEVICE, dtype=torch.float32)
test_slips_tensor = torch.from_numpy(test_slips).to(device=DEVICE, dtype=torch.float32)

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_phis_tensor))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(phis, observed_slips, "k.", label="Observations")
plt.plot(test_phis, mean.cpu().numpy(), "b", label="Predicted")
plt.fill_between(
    test_phis, lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5, color="blue"
)
plt.xlabel("Slope Angle (degrees)")
plt.ylabel("Slip Ratio")
plt.legend()
plt.show()
