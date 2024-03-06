"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction_models.trainers.regressor_trainer import RegressorTrainer
from src.prediction_models.trainers.utils import ParamsModelTraining


def main(device: str) -> None:
    """
    Main function for training the slip regressors.

    Parameters:
    - device (str): Device for training, either "cpu" or "cuda".
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

    # Initialize the regressor trainer
    trainer = RegressorTrainer(
        device=device,
        model_directory=model_directory,
        num_terrain_classes=10,
        slip_sensitivity_minmax=(1.0, 9.0),
        slip_nonlinearity_minmax=(1.4, 2.0),
        slip_offset_minmax=(0.0, 0.1),
        noise_scale_minmax=(0.1, 0.2),
        params_model_training=params_model_training,
    )

    # Train the slip regressors
    trainer.train_all_models()


if __name__ == "__main__":
    main("cuda:0")
