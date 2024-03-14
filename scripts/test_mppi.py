import sys
import os

# TODO: change the path adding method
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import fire

from gymnasium_env.env import PlanetaryExplorationEnv
from planner.mppi import MPPI


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PlanetaryExplorationEnv(seed=0, device=device)


if __name__ == "__main__":
    fire.Fire(main)
