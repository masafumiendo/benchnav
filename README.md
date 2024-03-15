# BenchNav
path/motion planning benchmark for planetary exploration rovers

## Dependencies

- NVIDIA Driver 510 or later (due to PyTorch 2.x)

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

## Test

```bash
cd /workspace/benchnav
python3 test/test_env.py
```
