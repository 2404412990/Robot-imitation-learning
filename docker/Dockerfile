FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=20 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    bzip2 \
    unzip \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    ffmpeg \
    xvfb \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libglfw3 \
    libglfw3-dev \
    libxi6 \
    libxrandr2 \
    libxxf86vm1 \
    libxinerama1 \
    libosmesa6 \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}
SHELL ["/bin/bash", "-lc"]

RUN conda config --set always_yes yes --set changeps1 no && \
    conda update -n base -c defaults conda && \
    conda create -n wham python=3.9 && \
    conda create -n gmr python=3.10 && \
    conda clean -afy

WORKDIR /opt/src/wham
COPY . /opt/src/wham

# WHAM runtime environment
RUN conda run -n wham pip install --upgrade "pip<26" "setuptools<81" "wheel<0.46" && \
        conda run -n wham pip install --index-url https://download.pytorch.org/whl/cu118 \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 && \
        conda run -n wham pip install --no-build-isolation -r requirements.txt && \
        conda run -n wham pip install json_tricks && \
    conda run -n wham pip install --no-deps --no-build-isolation -v third-party/ViTPose

# GMR runtime environment
RUN conda run -n gmr pip install --upgrade "pip<26" "setuptools<81" "wheel<0.46" && \
    conda run -n gmr pip install --index-url https://download.pytorch.org/whl/cu118 \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 && \
    conda run -n gmr pip install -e .

# Optional: if you need full WHAM demo.py SLAM dependencies, compile DPVO manually.
# Example:
#   conda run -n wham bash -lc "cd /workspace/third-party/DPVO && ... && pip install ."

ENV WHAM_PYTHON=/opt/conda/envs/wham/bin/python \
    GMR_PYTHON=/opt/conda/envs/gmr/bin/python \
    PYTHONPATH=/opt/src/wham

WORKDIR /workspace
CMD ["/bin/bash"]
