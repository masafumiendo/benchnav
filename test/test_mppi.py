"""
Masafumi Endo, 2024.
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from gpytorch.likelihoods import GaussianLikelihood

from src.environments.grid_map import GridMap
from src.simulator.planetary_env import PlanetaryEnv
from src.simulator.utils import ModelConfig
from src.simulator.robot_model import UnicycleModel
from src.planner.mppi import MPPI
from src.planner.objectives import Objectives
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.prediction_models.trainers.utils import load_model_state_dict
from src.prediction_models.terrain_classifiers.unet import Unet
from src.prediction_models.slip_regressors.gpr import GPModel
from src.prediction_models.traversability_predictors.classifier_and_regressor import (
    TraversabilityPredictor,
)


def main(device: str):

    # Set the data directory
    dataset_index = 1
    subset_index = 1
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(
        base_directory,
        f"datasets/dataset{dataset_index:02d}/test/subset{subset_index:02d}/000_000.pt",
    )

    # Load the tensor data
    data_item = torch.load(data_directory)
    tensors = data_item["tensors"]
    distributions = data_item["distributions"]

    # Set the terrain classifier
    num_total_terrain_classes = 10
    terrain_classifier = Unet(num_total_terrain_classes).to(device)

    # Set the model directory for the terrain classifier
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_directory = os.path.join(
        base_directory, f"trained_models/dataset{dataset_index:02d}/Unet/"
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

    terrain_classifier = load_model_state_dict(
        terrain_classifier, model_directory, device
    )

    # Set the model directory for the slip regressor
    model_directory = os.path.join(
        base_directory, f"trained_models/dataset{dataset_index:02d}/GPR/"
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
    train_data_directory = os.path.join(
        base_directory, f"datasets/dataset{dataset_index:02d}/slip_observations/"
    )

    all_gp_models = {}  # Dictionary to store the GP model
    all_train_data = {}  # Dictionary to store the training data
    for i in range(num_total_terrain_classes):
        # Load the training data
        train_data = torch.load(
            os.path.join(train_data_directory, f"{i:02d}_class.pth")
        )
        train_x = train_data["train_x"].to(device)
        train_y = train_data["train_y"].to(device)
        # Initialize the GP model
        likelihood = GaussianLikelihood().to(device=device)
        gp_model = GPModel(train_x, train_y, likelihood).to(device)
        # Load the trained model
        gp_model = load_model_state_dict(
            gp_model, os.path.join(model_directory, f"models/{i:02d}_class.pth"), device
        )
        # Store the model
        all_gp_models[i] = gp_model
        # Store the training data
        all_train_data[i] = train_data

    # Set the traversability predictor
    traversability_predictor = TraversabilityPredictor(
        terrain_classifier=terrain_classifier,
        slip_regressors=all_gp_models,
        device=device,
    )

    # Predict the traversability and store the predictive distribution
    predictions = traversability_predictor.predict(
        colors=tensors["colors"], slopes=tensors["slopes"]
    )
    distributions["predictions"] = predictions

    # Set the grid map
    grid_map = GridMap(
        grid_size=64, resolution=0.5, tensors=tensors, distributions=distributions
    )

    # Set the environment
    delta_t = 0.1
    time_limit = 100
    start_pos = torch.tensor([8, 8])
    goal_pos = torch.tensor([24, 24])
    env = PlanetaryEnv(
        grid_map=grid_map,
        start_pos=start_pos,
        goal_pos=goal_pos,
        seed=1,
        delta_t=delta_t,
        time_limit=time_limit,
        device=device,
    )

    # Set the unicycle model with the traversability model in inference mode
    model_config = ModelConfig(
        mode="inference", inference_metric="var", confidence_value=0.9
    )

    dynamics = UnicycleModel(
        grid_map=grid_map, model_config=model_config, device=device
    )

    # Set the objectives
    objectives = Objectives(dynamics, goal_pos=env._goal_pos)

    # Set the MPPI
    solver = MPPI(
        horizon=50,
        num_samples=10000,
        dim_state=3,
        dim_control=2,
        dynamics=dynamics.transit,
        stage_cost=objectives.stage_cost,
        terminal_cost=objectives.terminal_cost,
        u_min=env._dynamics._min_action,
        u_max=env._dynamics._max_action,
        sigmas=torch.tensor([0.5, 0.5]),
        lambda_=1.0,
    )

    state = env.reset(seed=0)
    max_steps = int(time_limit / delta_t)
    average_time = 0
    for i in range(max_steps):
        start = time.time()
        with torch.no_grad():
            action_seq, state_seq = solver.forward(state=state)
        end = time.time()
        average_time += (end - start) / max_steps

        state, is_terminated, is_truncated = env.step(action_seq[0, :])

        is_collisions = env.collision_check(states=state_seq)

        top_samples, top_weights = solver.get_top_samples(num_samples=1000)

        env.render(
            trajectory=state_seq,
            top_samples=(top_samples, top_weights),
            is_collisions=is_collisions,
        )
        if is_terminated:
            print("Goal Reached!")
            break

        if is_truncated:
            print("Time Limit Exceeded!")
            break


if __name__ == "__main__":
    main("cuda:0")
