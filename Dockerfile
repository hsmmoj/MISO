#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# System and build dependencies for OpenGL, Open3D, C++/CUDA, PyTorch3D, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        git \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        mesa-utils \
        x11-xserver-utils \
        binutils \
        build-essential \
        cmake \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*

# Install Miniconda (safe pinned version)
ENV CONDA_VERSION=py39_4.12.0
ENV CONDA_HOME=/opt/conda
ENV PATH=$CONDA_HOME/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_HOME && \
    rm /tmp/miniconda.sh

# Copy your environment file
COPY environment.yaml /tmp/environment.yaml

# Create the conda environment
RUN conda env create -f /tmp/environment.yaml && \
    conda clean -afy

# Set the default shell to use the conda environment
SHELL ["conda", "run", "-n", "miso", "/bin/bash", "-c"]

# Install PyTorch3D from source (best for compatibility)
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Default to bash shell in the environment
CMD ["/bin/bash"]
