"""
author: Masafumi Enod
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.prediction_models.terrain_classifiers.unet import Unet
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.prediction_models.trainers.utils import load_model_state_dict
from src.data.terrain_dataset import TerrainClassificationDataset as Dataset


def main(device: str) -> None:
    """
    Main function for testing the terrain classifier.

    Parameters:
    - device (str): Device for testing, either "cpu" or "cuda".
    """

    # Initialize the UNet model
    model = Unet(classes=10).to(device)
    # Set the data directory
    dataset_index = 1
    subset_index = 1
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(
        script_directory, f"datasets/dataset{dataset_index:02d}/"
    )
    # Set the model directory
    model_directory = os.path.join(
        script_directory, f"trained_models/dataset{dataset_index:02d}/Unet/"
    )
    # Set the parameters for model training
    params_model_training = ParamsModelTraining(
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=0e00,
        num_epochs=100,
        save_interval=None,
        device=device,
    )
    # Load the trained model
    model_directory = os.path.join(
        model_directory,
        f"bs{params_model_training.batch_size:03d}_"
        f"lr{params_model_training.learning_rate:.0e}_"
        f"wd{params_model_training.weight_decay:.0e}_"
        f"epochs{params_model_training.num_epochs:03d}",
        "models/best_model.pth",
    )

    model = load_model_state_dict(model, model_directory, device)

    # Load the test dataset
    test_dataset = Dataset(data_directory, "test", subset_index)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Test the model
    for _, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        predicted = model.predict(inputs)

        # Data conversion for visualization
        inputs = inputs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        targets = targets.squeeze(0).cpu().numpy()
        predicted = predicted.squeeze(0).cpu().detach().numpy()

        # Visualization
        _, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Input Image
        axs[0].imshow(inputs)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Ground Truth
        axs[1].imshow(targets, cmap="jet")
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        # Predicted
        axs[2].imshow(predicted, cmap="jet")
        axs[2].set_title("Predicted")
        axs[2].axis("off")

        plt.show()


if __name__ == "__main__":
    main("cuda:0")
