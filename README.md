<h1 align="center">
BenchNav 
</h1>

BenchNav is a Pytorch-based simulation platform designed for **Bench**marking off-road **Nav**igation algorithms.
On top of Gymnasium, we implement the simulation platform owing the following features:

- **Synthetic Terrain Data Generation**: generates a top-down 2.5D terrain map instance replicating pixel-wise appearance and geometric features with their corresponding latent traversability coefficients.
- **Probabilistic Traversability Prediction**: employs built-in ML models to construct probability distributions of traversability coefficients.
- **Path and Motion Planning Execution**: simulates off-road point-to-goal navigation by 1) defining motion planning problems and 2) deploying solvers, such as path and motion planners, to find a sequence of feasible actions in an iterative fashion.

### Representative Off-road Navigation Examples

| ![A* + DWA](/assets/AStar_DWA.gif)  A* + DWA | ![CL-RRT](/assets/CL_RRT.gif)  CL-RRT | ![MPPI](/assets/MPPI.gif)  MPPI |
|:---:|:---:|:---:|

Trajectories are color-coded according to traversability, with cooler colors for safer paths.

### Project Structure at a Glance
```
├── src                         # Source code directory
│   ├── data                    # Synthetic terrain dataset generation
│   │   └── dataset_generator.py
│   ├── environments            # Grid map definitions for environmental features
│   │   └── grid_map.py
│   ├── planners                # Example planners implementation
│   │   ├── global_planners
│   │   └── local_planners
│   ├── prediction_models       # Probabilistic traversability prediction for ML models
│   │   ├── slip_regressors
│   │   ├── terrain_classifiers
│   │   └── traversability_predictors
│   ├── simulator               # Gym-based simulator for off-road navigation
│   │   ├── problem_formulation
│   │   └── planetary_env.py
│   └── utils                   # Utility scripts and helper functions
├── datasets                    # Datasets for model training and evaluation
├── notebooks                   # Notebooks containing tutorials for the platform
└── trained_models              # Pretrained ML Models

```

## Citation

```
@INPROCEEDINGS{endo2024benchnav, 
  AUTHOR    = {Masafumi Endo and Kohei Honda and Genya Ishigami}, 
  TITLE     = {BenchNav: Simulation Platform for Benchmarking Off-road Navigation Algorithms with Probabilistic Traversability}, 
  BOOKTITLE = {under review for ICRA 2024 Workshop on Resilient Off-road Autonomy}, 
  YEAR      = {2024}, 
  ADDRESS   = {Yokohama, Japan}, 
  MONTH     = {June}
} 
```

## Dependencies

- NVIDIA Driver 510 or later (due to PyTorch 2.x) if you want to use GPU

## Installation

<details>
<summary>Docker Installation</summary>

### Install Docker

[Installation guide](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

```bash
# Install from get.docker.com
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo groupadd docker
sudo usermod -aG docker $USER
```

### Setup GPU for Docker
[Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list 

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```
</details>

### Setup with Docker

```bash
# build container (with GPU support)
make build-gpu
# or build container (without GPU support)
# make build-cpu

# Open remote container via Vscode (Recommend)
# 1. Open the folder using vscode
# 2. Ctrl+P and select 'devcontainer rebuild and reopen in container'
# Then, you can skip the following commands

# Or Run container via terminal (with GPU support)
make bash-gpu
# or Run container via terminal (without GPU support)
# make bash-cpu
```

## Usage

### Dataset and ML models download

```bash
cd /workspace/benchnav
sh scripts/download_dataset_and_model.sh
```

### (Optional) Dataset preparation and ML models training
- To generate a terrain dataset:
```bash
cd /workspace/benchnav
python3 scripts/generate_terrain_dataset.py
```

- To train the terrain classifier:
```bash
cd /workspace/benchnav
python3 scripts/train_terrain_classifier.py
```

- To train the slip regressors:
```bash
cd /workspace/benchnav
python3 scripts/train_slip_regressors.py
```

### Off-road Navigation Simulation

You can see step-by-step instructions for simulating off-road navigations at Tutorial #3.X.

## Tutorials
- Tutorial #1: Environment Descriptions
- Tutorial #2: Traversability Prediction Models
- Tutorial #3.1: Off-road Navigation with A* + DWA
- Tutorial #3.2: Off-road Navigation with CL-RRT
- Tutorial #3.3: Off-road Navigation with MPPI