FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install --no-install-recommends\
    software-properties-common\
    libgl1-mesa-dev\
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    wget\
    libssl-dev\
    curl\
    git\
    x11-apps \
    swig \
    patchelf

# Python (version 3.10)
RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && apt-get install -y \
  python3.10 \
  python3.10-dev \
  python3.10-venv \
  python3.10-distutils \
  python3.10-tk

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN pip3 install --upgrade pip
RUN pip3 install -U pip distlib setuptools wheel

# utils
RUN apt-get update && apt-get install -y htop vim ffmpeg 
# RUN pip3 install jupyterlab ipywidgets && \
#     echo 'alias jup="jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &"' >> /root/.bashrc

# clear cache
RUN rm -rf /var/lib/apt/lists/*

# pytorch 2.0
RUN pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
COPY src/ src/
COPY pyproject.toml .

RUN pip3 install -e .[dev]

CMD ["bash"]
