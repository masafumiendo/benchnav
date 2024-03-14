"""
Masafumi Endo, 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from src.environments.slip_model import SlipModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate the SlipModel
model = SlipModel(
    slip_sensitivity=1.0, slip_nonlinearity=2.0, slip_offset=0.1, device=DEVICE
)

# Create an array of slope angles (phi)
phis = torch.linspace(0, 30, 500).to(device=DEVICE)  # Slope angles ranging from 0 to 30

# Obtain distributions of slip values from the slope angles
slip_dist = model.model_distribution(phis)

# Retrieve the mean and standard deviation of the slip distributions
mean_slip = slip_dist.mean.cpu().numpy()
std_slip = slip_dist.stddev.cpu().numpy()
observed_slips = slip_dist.sample().cpu().numpy()

# Compute dynamic noise scales for the visualization
# Assuming the model has methods or attributes to compute or retrieve the dynamic noise scale for each phi
lowers = mean_slip - 2 * std_slip  # Assuming 2 standard deviations for the bounds
uppers = mean_slip + 2 * std_slip

# Visualize the results with dynamic noise bounds
plt.figure(figsize=(10, 6))
plt.plot(phis.cpu().numpy(), mean_slip, label="Actual Slip", color="blue")
plt.fill_between(
    phis.cpu().numpy(),
    lowers,
    uppers,
    color="skyblue",
    alpha=0.4,
    label="Confidence Interval (Noise Region)",
)
plt.scatter(
    phis.cpu().numpy(),
    observed_slips,
    color="red",
    s=10,
    label="Observed Slip",
    alpha=0.6,
)
plt.xlim(0, 30)
plt.ylim(0, 1)
plt.xlabel("Slope Angle (phi)")
plt.ylabel("Slip Ratio")
plt.title("Slip Ratio vs. Slope Angle with Heteroscedastic Noise")
plt.legend()
plt.grid(True)
plt.show()
