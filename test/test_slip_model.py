"""
author: Masafumi Endo
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
    device=DEVICE, slip_sensitivity=1.0, slip_nonlinearity=2.0, slip_offset=0.1
)

# Create an array of slope angles (phi)
phis = torch.linspace(-30, 30, 500).to(
    device=DEVICE
)  # Slope angles ranging from -30 to 30

# Compute the observed slip values
observed_slips = model.observe_slip(phis).cpu().numpy()
actual_slips = model.latent_model(phis).cpu().numpy()

# Compute dynamic noise scales for the visualization
# Assuming the model has methods or attributes to compute or retrieve the dynamic noise scale for each phi
noise_scales = model.noise_model(phis)
lowers = (
    actual_slips - 2 * noise_scales.cpu().numpy()
)  # Assuming 2 standard deviations for the bounds
uppers = actual_slips + 2 * noise_scales.cpu().numpy()

# Visualize the results with dynamic noise bounds
plt.figure(figsize=(10, 6))
plt.plot(phis.cpu().numpy(), actual_slips, label="Actual Slip", color="blue")
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
plt.xlim(-30, 30)
plt.ylim(-1, 1)
plt.xlabel("Slope Angle (phi)")
plt.ylabel("Slip Ratio")
plt.title("Slip Ratio vs. Slope Angle with Heteroscedastic Noise")
plt.legend()
plt.grid(True)
plt.show()

phi = 10.0
observed_slip = model.observe_slip(phi)
