# POLAR — multi-task VRP with neural decoding + PyVRP local search
# GPU image: Python 3.10, PyTorch 2.0.1, CUDA 11.8

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/AJ-Correa/Routing-POLAR"
LABEL org.opencontainers.image.description="Improving Cross-Problem Vehicle Routing with Locally Augmented Preferences and Representation Disentanglement"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.10 + build tools (Cython / PyVRP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1) Pin CUDA 11.8 PyTorch before anything that declares torch>=...
RUN pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 2) Remaining requirements except torch* and packages that must use --no-deps
RUN grep -vE '^(#|$|torch==|torchvision==|torchaudio==|rl4co==|torchrl==|lion-pytorch==)' requirements.txt \
        > /tmp/requirements-base.txt \
    && pip install -r /tmp/requirements-base.txt

# 3) torch-sensitive packages: install without resolving their torch deps
RUN pip install rl4co==0.2.0 torchrl==0.1.1 lion-pytorch==0.2.4 --no-deps

COPY . .

RUN cd search/cython_heuristics && python setup.py build_ext --inplace

RUN python -c "import torch; assert torch.__version__.startswith('2.0.1'), torch.__version__"

CMD ["python", "run.py", "--help"]
