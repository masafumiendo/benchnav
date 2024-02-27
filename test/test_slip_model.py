"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from src.environments.slip_model import SlipModel

# Instantiate the SlipModel
model = SlipModel(slip_sensitivity=1.0, slip_nonlinearity=2.0, slip_offset=0.1)

# Create an array of slope angles (phi)
phis = torch.linspace(-30, 30, 500)  # Slope angles ranging from -10 to 10

# Compute the observed slip values
observed_slips = model.observe_slip(phis).detach().numpy()
actual_slips = model.latent_model(phis).detach().numpy()

lowers = actual_slips - 2 * model.noise_scale
uppers = actual_slips + 2 * model.noise_scale

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(phis, actual_slips, label="Actual Slip")
plt.fill_between(phis, lowers, uppers, alpha=0.3)
plt.scatter(phis, observed_slips, label="Observed Slip")
plt.xlabel("Slope Angle (phi)")
plt.ylabel("Slip Ratio")
plt.title("Slip Ratio vs. Slope Angle")
plt.legend()
plt.grid(True)
plt.show()
