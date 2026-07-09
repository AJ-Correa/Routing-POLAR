# POLAR — multi-task VRP with neural decoding + PyVRP local search
# GPU image: Python 3.10, PyTorch 2.0.1, CUDA 11.8

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/AJ-Correa/Routing-POLAR"
LABEL org.opencontainers.image.description="Improving Cross-Problem Vehicle Routing with Locally Augmented Preferences and Representation Disentanglement"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.10 + compilers (needed for Cython extension and PyVRP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app  # container project root; bind-mount host repo here at runtime

# PyTorch first — pin before any package that declares torch>=...
RUN pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .

# Base deps (no torch pin upgrades)
RUN pip install \
    numpy==1.24.4 \
    scipy==1.15.3 \
    Cython==3.2.4 \
    einops==0.8.2 \
    pyvrp==0.11.0 \
    vrplib==1.5.1 \
    wandb==0.26.0 \
    PyYAML==6.0.3 \
    openpyxl==3.1.5 \
    pandas==2.3.3 \
    matplotlib==3.10.8 \
    rich==15.0.0 \
    tqdm==4.67.3 \
    pydantic==2.13.3 \
    requests==2.28.1

# Order matters — rl4co/torchrl/lion-pytorch would otherwise pull torch 2.13+
RUN pip install tensordict==0.1.2 \
    && pip install rl4co==0.2.0 --no-deps \
    && pip install torchrl==0.1.1 --no-deps \
    && pip install lion-pytorch==0.2.4 --no-deps

# Application source
COPY . .

# Build Cython heuristics in-place
RUN cd search/cython_heuristics && python setup.py build_ext --inplace

# Sanity check: correct torch + CUDA
RUN python -c "import torch; assert torch.__version__.startswith('2.0.1'), torch.__version__"

CMD ["python", "run.py", "--help"]
