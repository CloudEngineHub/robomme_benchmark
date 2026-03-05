FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    gnupg \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglvnd-dev \
    libglib2.0-0 \
    libvulkan1 \
    vulkan-tools \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3 \
    && rm -f /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OMP_NUM_THREADS=1 \
    PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt pyproject.toml README.md ./
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r requirements.txt

COPY --chown=user:user . .
RUN python3 -m pip install -e .
RUN chmod +x /home/user/app/docker-entrypoint.sh

USER user
EXPOSE 7860
ENTRYPOINT ["./docker-entrypoint.sh"]

CMD ["python3", "gradio-web/main.py"]
