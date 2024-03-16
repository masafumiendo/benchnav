"""
Masafumi Endo, 2024.
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from src.environments.grid_map import GridMap
from src.simulator.planetary_env import PlanetaryEnv
from src.planner.mppi import MPPI
from src.planner.objectives import Objectives


def main():

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
    )

    # Set the objectives
    objectives = Objectives(grid_map=grid_map, goal_pos=env._goal_pos)

    # Set the MPPI
    solver = MPPI(
        horizon=50,
        num_samples=10000,
        dim_state=3,
        dim_control=2,
        dynamics=env._dynamics.transit,
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
    main()
