ARG PYTORCH="2.3.1"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# avoid selecting 'Geographic area' during installation
ARG DEBIAN_FRONTEND=noninteractive

# apt install required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    git wget sudo htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip==24.0

# Install JAX with CUDA 12 support (required for waymax, flax, optax, chex)
RUN pip install --no-cache-dir \
    "jax[cuda12]==0.4.30" \
    jaxlib==0.4.30

# Install TensorFlow (CUDA support comes bundled in 2.17)
RUN pip install --no-cache-dir \
    tensorflow==2.17.0 \
    tensorboard==2.17.0

# Install PyTorch Lightning and related
RUN pip install --no-cache-dir \
    lightning==2.3.3 \
    pytorch-lightning==2.3.3 \
    lightning-utilities==0.11.5 \
    torchmetrics==1.4.0.post0

# Install JAX ecosystem
RUN pip install --no-cache-dir \
    flax==0.8.5 \
    optax==0.2.3 \
    chex==0.1.86 \
    orbax-checkpoint==0.5.20 \
    tensorstore==0.1.63

# Install DeepMind utilities
RUN pip install --no-cache-dir \
    dm-env==1.6 \
    dm-tree==0.1.8

# Install Keras
RUN pip install --no-cache-dir keras==3.4.1

# Install scientific computing and visualization
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.14.0 \
    matplotlib==3.9.1 \
    pillow==10.4.0 \
    imageio==2.34.2 \
    mediapy==1.2.2 \
    h5py==3.11.0

# Install Jupyter/IPython ecosystem
RUN pip install --no-cache-dir \
    ipykernel==6.29.5 \
    ipython==8.26.0 \
    jupyter-client==8.6.2 \
    jupyter-core==5.7.2

# Install experiment tracking
RUN pip install --no-cache-dir \
    wandb==0.17.5 \
    tqdm==4.66.4 \
    rich==13.7.1

# Install remaining utilities
RUN pip install --no-cache-dir \
    absl-py==2.1.0 \
    immutabledict==4.2.0 \
    etils==1.7.0 \
    protobuf==4.25.3

RUN pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax

# Default command
CMD ["/bin/bash"]