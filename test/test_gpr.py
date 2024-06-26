"""
Masafumi Endo, 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gpytorch
import matplotlib.pyplot as plt

from src.environments.slip_model import SlipModel
from src.prediction_models.slip_regressors.gpr import GPModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate the SlipModel
slip_model = SlipModel(
    slip_sensitivity=1.0, slip_nonlinearity=2.0, slip_offset=0.1, device=DEVICE
)

# Generate synthetic data
phis = torch.linspace(0, 30, 1000).to(device=DEVICE)
slip_dist = slip_model.model_distribution(phis)

# Sample observed slips
observed_slips = slip_dist.sample()

# Instantiate the GPModel
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=DEVICE)
model = GPModel(phis, observed_slips, likelihood).to(device=DEVICE)

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
    output = model(phis)
    # Calc loss and backprop derivatives
    loss = -mll(output, observed_slips)
    loss.backward()
    print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item()}")
    optimizer.step()

# Generate test data
test_phis = torch.linspace(0, 30, 100).to(device=DEVICE)
test_slips = slip_model.model_distribution(test_phis).sample()

# Make predictions
pred_dist = model.predict(test_phis)
mean = pred_dist.mean
lower, upper = pred_dist.confidence_region()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(phis.cpu().numpy(), observed_slips.cpu().numpy(), "k.", label="Observations")
plt.plot(test_phis.cpu().numpy(), mean.cpu().numpy(), "b", label="Predicted")
plt.fill_between(
    test_phis.cpu().numpy(),
    lower.cpu().numpy(),
    upper.cpu().numpy(),
    alpha=0.5,
    color="blue",
)
plt.xlim(0, 30)
plt.ylim(0, 1)
plt.xlabel("Slope Angle (degrees)")
plt.ylabel("Slip Ratio")
plt.legend()
plt.show()
