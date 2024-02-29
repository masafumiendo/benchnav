"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction_models.trainers.classifier_trainer import ClassifierTrainer
from src.prediction_models.terrain_classifiers.unet import Unet
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.data.terrain_dataset import TerrainDataset


def main(device: str) -> None:
    """
    Main function for training the terrain classifier.

    Parameters:
    - device (str): Device for training, either "cpu" or "cuda".
    """

    # Initialize the UNet model
    model = Unet(classes=10).to(device)
    # Set the data directory
    dataset_index = 1
    subset_index = 1
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(
        script_directory,
        f"datasets/dataset{dataset_index:02d}/subset{subset_index:02d}/",
    )
    # Set the model directory
    model_directory = os.path.join(
        script_directory,
        f"trained_models/dataset{dataset_index:02d}/subset{subset_index:02d}/Unet/",
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

    # Load the training and validation datasets
    train_dataset = TerrainDataset(data_directory, "train")
    valid_dataset = TerrainDataset(data_directory, "valid")

    # Initialize the classifier trainer
    trainer = ClassifierTrainer(
        model=model,
        model_directory=model_directory,
        params_model_training=params_model_training,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main("cuda:1")
