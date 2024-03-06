"""
author: Masafumi Endo
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module

from src.prediction_models.trainers.utils import ParamsModelTraining
from src.data.terrain_dataset import TerrainDataset


class ClassifierTrainer:
    def __init__(
        self,
        model: Module,
        model_directory: str,
        params_model_training: ParamsModelTraining,
        train_dataset: TerrainDataset,
        valid_dataset: Optional[TerrainDataset] = None,
    ) -> None:
        """
        Initialize ClassifierTrainer class.

        Parameters:
        - model (Module): Model to be trained.
        - model_directory (str): Directory to save model checkpoints and logs.
        - params_model_training (ParamsModelTraining): Parameters for model training.
        - train_dataset (TerrainDataset): Training dataset.
        - valid_dataset (Optional[TerrainDataset]): Validation dataset.
        """
        self.model = model
        self.model_directory = model_directory
        self.params_model_training = params_model_training
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # Initialize data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=params_model_training.batch_size, shuffle=True
        )
        self.valid_loader = (
            DataLoader(
                valid_dataset,
                batch_size=params_model_training.batch_size,
                shuffle=False,
            )
            if valid_dataset is not None
            else None
        )

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=params_model_training.learning_rate,
            weight_decay=params_model_training.weight_decay,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Create model directory if it does not exist
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        ymdhms = datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )  # Year, month, day, hour, minute, second
        self.model_directory = os.path.join(
            model_directory,
            f"bs{params_model_training.batch_size:03d}_"
            f"lr{params_model_training.learning_rate:.0e}_"
            f"wd{params_model_training.weight_decay:.0e}_"
            f"epochs{params_model_training.num_epochs:03d}",
        )
        self.model_saving_path = os.path.join(self.model_directory, "models")
        self.log_saving_path = os.path.join(self.model_directory, f"logs/{ymdhms}")

        # Create model and log directories if they do not exist
        if not os.path.exists(self.model_saving_path):
            os.makedirs(self.model_saving_path)
        if not os.path.exists(self.log_saving_path):
            os.makedirs(self.log_saving_path)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_saving_path)

    def train(self) -> None:
        """
        Train the model with the training dataset.
        """
        best_valid_loss = float("inf")

        for epoch in range(self.params_model_training.num_epochs):
            self.model.train()
            total_train_loss = 0.0

            # Iterate over the training dataset
            for index, (x, y) in enumerate(self.train_loader):
                x, y = (
                    x.to(self.params_model_training.device),
                    y.to(self.params_model_training.device),
                )
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                self.writer.add_scalar(
                    "Loss/train", loss.item(), epoch * len(self.train_loader) + index
                )

            # Log average training loss
            average_train_loss = total_train_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/train/average", average_train_loss, epoch)

            # Log validation loss if validation dataset is provided
            if self.valid_loader is not None:
                total_valid_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for x, y in self.valid_loader:
                        x, y = (
                            x.to(self.params_model_training.device),
                            y.to(self.params_model_training.device),
                        )
                        y_pred = self.model(x)
                        loss = self.loss_fn(y_pred, y)
                        total_valid_loss += loss.item()
                average_valid_loss = total_valid_loss / len(self.valid_loader)
                self.writer.add_scalar("Loss/valid", average_valid_loss, epoch)

                print(
                    f"Epoch {epoch+1}/{self.params_model_training.num_epochs}, "
                    f"Train Loss: {average_train_loss:.4f}, "
                    f"Valid Loss: {average_valid_loss:.4f}"
                )

                # Save model checkpoint if validation loss is improved
                if average_valid_loss < best_valid_loss:
                    best_valid_loss = average_valid_loss
                    self.save_model()
            else:
                print(
                    f"Epoch {epoch+1}/{self.params_model_training.num_epochs}, "
                    f"Train Loss: {average_train_loss:.4f}"
                )

    def save_model(self) -> None:
        """
        Save the model checkpoint.
        """
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_saving_path, "best_model.pth"),
        )
        print(f"Model saved at {self.model_saving_path}")
