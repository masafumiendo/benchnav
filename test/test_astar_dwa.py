"""
Masafumi Endo, 2024.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.environments.grid_map import GridMap
from src.simulator.planetary_env import PlanetaryEnv
from src.simulator.utils import ModelConfig
from src.simulator.robot_model import UnicycleModel
from src.planners.local_planners.objectives import Objectives
from src.planners.local_planners.dwa import DWA
from src.planners.global_planners.astar import AStar
from src.prediction_models.trainers.utils import ParamsModelTraining
from src.prediction_models.trainers.utils import load_model_state_dict
from src.prediction_models.trainers.utils import load_slip_regressors
from src.prediction_models.terrain_classifiers.unet import Unet
from src.prediction_models.slip_regressors.gpr import GPModel
from src.prediction_models.traversability_predictors.classifier_and_regressor import (
    TraversabilityPredictor,
)


def main(device: str):

    # Set the data directory
    dataset_index = 1
    subset_index = 1
    instance_name = "000_000"
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(
        base_directory,
        f"datasets/dataset{dataset_index:02d}/test/subset{subset_index:02d}/{instance_name}.pt",
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
        base_directory, f"datasets/dataset{dataset_index:02d}/"
    )

    all_gp_models = load_slip_regressors(
        model=GPModel,
        num_terrain_classes=num_total_terrain_classes,
        model_directory=model_directory,
        train_data_directory=train_data_directory,
        device=device,
    )

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
        grid_size=64,
        resolution=0.5,
        tensors=tensors,
        distributions=distributions,
        instance_name=instance_name,
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
        stuck_threshold=0.3,
        device=device,
    )

    # Set the unicycle model with the traversability model in inference mode
    model_config = ModelConfig(
        mode="inference", inference_metric="cvar", confidence_value=0.9
    )

    dynamics = UnicycleModel(
        grid_map=grid_map, model_config=model_config, device=device
    )

    # Initialize the DWA local planner and the objectives
    objectives = Objectives(
        dynamics=dynamics, goal_pos=env._goal_pos, stuck_threshold=env.stuck_threshold
    )

    solver = DWA(
        horizon=50,
        dim_state=3,
        dim_control=2,
        dynamics=dynamics.transit,
        stage_cost=objectives.stage_cost,
        terminal_cost=objectives.terminal_cost,
        u_min=dynamics.min_action,
        u_max=dynamics.max_action,
        a_lim=torch.tensor([0.5, 0.5]),
        delta_t=delta_t,
        num_lin_vel=10,
        num_ang_vel=10,
    )

    # Initialize the A* global planner
    astar = AStar(
        grid_map=grid_map,
        goal_pos=goal_pos,
        dynamics=dynamics,
        stuck_threshold=env.stuck_threshold,
    )

    state = env.reset(seed=0)
    for _ in range(int(time_limit / delta_t)):
        with torch.no_grad():
            reference_path = astar.forward(state=state)
            solver.update_reference_path(reference_path)
            action_seq, state_seq = solver.forward(state=state)

        state, reward, is_terminated, is_truncated = env.step(action_seq[0, :])

        is_collisions = env.collision_check(states=state_seq)

        top_samples, top_weights = solver.get_top_samples()

        env.render(
            trajectory=state_seq,
            top_samples=(top_samples, top_weights),
            is_collisions=is_collisions,
            reference_path=solver._reference_path,
        )

        if is_terminated:
            print("Goal Reached!")
            break

        if is_truncated:
            print("Time Limit Exceeded!")
            break


if __name__ == "__main__":
    main("cuda:0")
