"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import matplotlib.pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood

from src.prediction_models.slip_regressors.gpr import GPModel
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.prediction_models.trainers.utils import load_model_state_dict


def main(device: str) -> None:
    """
    Main function for testing the slip regressors.

    Parameters:
    - device (str): Device for testing, either "cpu" or "cuda".
    """

    # Set the model directory
    dataset_index = 1
    subset_index = 1
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_directory = os.path.join(
        script_directory,
        f"trained_models/dataset{dataset_index:02d}/subset{subset_index:02d}/GPR/",
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

    likelihood = GaussianLikelihood().to(device=device)

    # Set the test inputs
    test_phis = torch.linspace(-30, 30, 100).to(device=device)

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
        model = GPModel(train_x, train_y, likelihood).to(device)

        # Load the trained model
        model = load_model_state_dict(
            model=model,
            model_directory=os.path.join(model_directory, f"models/{i:02d}_class.pth"),
            device=device,
        )

        # Test the trained model
        mean, lower, upper = model.predict(test_phis)

        # Load the actual model for reference
        with open(os.path.join(data_directory, f"models/{i:02d}_class.pkl"), "rb") as f:
            slip_model = pickle.load(f)
        test_slips = slip_model.observe_slip(test_phis)

        # Plot the results
        # Pytorch -> Numpy conversion
        test_phis = test_phis.cpu().detach().numpy()
        mean = mean.cpu().detach().numpy()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        test_slips = test_slips.cpu().detach().numpy()

        # Plot the results in a subplot
        ax = axes.flatten()[i]
        ax.plot(test_phis, mean, "r", label="Predicted Mean")
        ax.fill_between(
            test_phis, lower, upper, color="r", alpha=0.5, label="Confidence"
        )
        ax.plot(test_phis, test_slips, "k.", label="Actual Slip")
        ax.set_title(f"Terrain {i+1}")
        ax.set_xlabel("Phi")
        ax.set_ylabel("Slip")
        ax.legend(loc="upper left")

        # back to torch
        test_phis = torch.from_numpy(test_phis).to(device=device)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main("cuda:0")
