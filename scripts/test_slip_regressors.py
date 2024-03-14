"""
Masafumi Endo, 2024
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from gpytorch.likelihoods import GaussianLikelihood

from src.prediction_models.slip_regressors.gpr import GPModel
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.prediction_models.trainers.utils import load_model_state_dict
from src.data.slip_models_generator import SlipModelsGenerator

sns.set()


def main(device: str) -> None:
    """
    Main function for testing the slip regressors.

    Parameters:
    - device (str): Device for testing, either "cpu" or "cuda".
    """

    # Set the model directory
    dataset_index = 1
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_directory = os.path.join(
        script_directory, f"trained_models/dataset{dataset_index:02d}/GPR/"
    )
    # Set the parameters for model training
    params_model_training = ParamsModelTraining(
        learning_rate=1e-1, num_iterations=100, device=device
    )
    model_directory = os.path.join(
        model_directory,
        f"lr{params_model_training.learning_rate:.0e}_"
        f"iters{params_model_training.num_iterations:03d}",
    )
    # Set the data directory
    data_directory = os.path.join(
        script_directory, f"datasets/dataset{dataset_index:02d}/slip_models/"
    )

    # Set the test inputs
    test_phis = torch.linspace(0, 30, 100).to(device=device)

    # Load the actual model for reference
    slip_sensitivity_minmax = (1.0, 9.0)
    slip_nonlinearity_minmax = (1.4, 2.0)
    slip_offset_minmax = (0.0, 0.1)
    noise_scale_minmax = (0.1, 0.2)

    # Generate the slip models
    slip_models_generator = SlipModelsGenerator(
        num_total_terrain_classes=10,
        slip_sensitivity_minmax=slip_sensitivity_minmax,
        slip_nonlinearity_minmax=slip_nonlinearity_minmax,
        slip_offset_minmax=slip_offset_minmax,
        noise_scale_minmax=noise_scale_minmax,
        device=device,
    )
    slip_models = slip_models_generator.generate_slip_models()

    # Prepare figure for subplots
    fig, axes = plt.subplots(
        nrows=2, ncols=5, figsize=(20, 8)
    )  # Adjust nrows and ncols based on your preference
    fig.suptitle("GPR Predictions vs. Actual Slip Models")

    # Load all the trained and actual models and test them
    for i in range(10):
        # Load the training data
        train_data = torch.load(
            os.path.join(data_directory, f"observations/{i:02d}_class.pth")
        )
        train_x = train_data["train_x"].to(device=device)
        train_y = train_data["train_y"].to(device=device)
        # Initialize the GP model
        likelihood = GaussianLikelihood().to(device=device)
        model = GPModel(train_x, train_y, likelihood).to(device)

        # Load the trained model
        model = load_model_state_dict(
            model=model,
            model_directory=os.path.join(model_directory, f"models/{i:02d}_class.pth"),
            device=device,
        )

        # Test the trained model
        pred_dist = model.predict(test_phis)
        pred_mean = pred_dist.mean
        pred_std = pred_dist.stddev
        # Get the lower and upper bounds of the confidence interval
        pred_lower, pred_upper = pred_mean - 2 * pred_std, pred_mean + 2 * pred_std

        # Retrieve the actual slip model
        slip_dist = slip_models[i].model_distribution(test_phis)
        slip_mean = slip_dist.mean
        slip_std = slip_dist.stddev
        # Get the lower and upper bounds of the confidence interval
        slip_lower, slip_upper = slip_mean - 2 * slip_std, slip_mean + 2 * slip_std

        # Plot the results
        # Pytorch -> Numpy conversion
        test_phis = test_phis.cpu().detach().numpy()
        pred_mean = pred_mean.cpu().detach().numpy()
        pred_lower = pred_lower.cpu().detach().numpy()
        pred_upper = pred_upper.cpu().detach().numpy()
        slip_mean = slip_mean.cpu().detach().numpy()
        slip_lower = slip_lower.cpu().detach().numpy()
        slip_upper = slip_upper.cpu().detach().numpy()
        train_x = train_x.cpu().detach().numpy()
        train_y = train_y.cpu().detach().numpy()

        # Plot the results in a subplot
        ax = axes.flatten()[i]
        # Actual model
        ax.plot(test_phis, slip_mean, "b", label="Actual Model")
        ax.fill_between(
            test_phis,
            slip_lower,
            slip_upper,
            color="b",
            alpha=0.5,
            label="Actual Confidence",
        )
        # GP model
        ax.plot(test_phis, pred_mean, "r", label="Predicted Mean")
        ax.fill_between(
            test_phis,
            pred_lower,
            pred_upper,
            color="r",
            alpha=0.5,
            label="Predicted Confidence",
        )
        # Training data
        ax.scatter(train_x, train_y, color="r", label="Training Data", s=1)
        ax.set_title(f"Terrain {i+1}")
        ax.set_xlabel("Phi")
        ax.set_ylabel("Slip")
        ax.set_xlim([0, 30])
        ax.set_ylim([0, 1])

        # back to torch
        test_phis = torch.from_numpy(test_phis).to(device=device)

    # Custom legend entries
    legend_elements = [
        Line2D([0], [0], color="b", label="Actual Model"),
        Line2D([0], [0], color="b", alpha=0.5, label="Actual Confidence", linewidth=10),
        Line2D([0], [0], color="r", label="Predicted Mean"),
        Line2D(
            [0], [0], color="r", alpha=0.5, label="Predicted Confidence", linewidth=10
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Training Data",
            markerfacecolor="r",
            markersize=5,
        ),
    ]

    # Add the global legend
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.05)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main("cuda:0")
