"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction_models.trainers.regressor_trainer import RegressorTrainer
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.data.terrain_dataset import SlipRegressionDataset as Dataset


def main(device: str) -> None:
    """
    Main function for training the slip regressors.

    Parameters:
    - device (str): Device for training, either "cpu" or "cuda".
    """

    # Set the data directory
    dataset_index = 1
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(
        script_directory, f"datasets/dataset{dataset_index:02d}/"
    )
    # Set the model directory
    model_directory = os.path.join(
        script_directory, f"trained_models/dataset{dataset_index:02d}/GPR/"
    )

    # Set the parameters for model training
    params_model_training = ParamsModelTraining(
        batch_size=16, learning_rate=1e-1, num_iterations=100, device=device
    )

    # Load the training dataset
    train_dataset = Dataset(data_directory, "train")

    # Initialize the regressor trainer
    trainer = RegressorTrainer(
        device=device,
        model_directory=model_directory,
        data_directory=data_directory,
        num_terrain_classes=10,
        params_model_training=params_model_training,
        train_dataset=train_dataset,
    )

    # Train the slip regressors
    trainer.train_all_models()


if __name__ == "__main__":
    main("cuda:0")
