# OpenVLA + LIBERO reproduction environment
# Pins match the OpenVLA README's reported setup EXACTLY. Do not "upgrade" anything here:
# OpenVLA ships custom modeling code that breaks on transformers > 4.40.1.
#
# Base: CUDA 12.1 devel (nvcc required to build flash-attn), Ubuntu 22.04 -> Python 3.10.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # MuJoCo headless rendering on a server with no display.
    # EGL uses the GPU; this is the setting that makes or breaks step 1.2.
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    HF_HOME=/cache/hf

# --- System deps -------------------------------------------------------------
# libegl1 / libglew / libosmesa: MuJoCo rendering backends (EGL primary, OSMesa fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential \
      python3.10 python3.10-dev python3-pip \
      libegl1 libegl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
      libglfw3 libglib2.0-0 patchelf \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && python -m pip install --upgrade pip setuptools wheel

# --- PyTorch first -----------------------------------------------------------
# flash-attn compiles against the installed torch, so torch must land before it.
RUN pip install --no-cache-dir \
      torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# --- flash-attn 2.5.5 --------------------------------------------------------
# Building from source takes 30-60+ min on 4 vCPU. Prefer the prebuilt wheel.
# MAX_JOBS caps RAM use; a parallel build OOMs a 16 GB box.
ENV MAX_JOBS=4
RUN pip install --no-cache-dir ninja packaging && \
    pip install --no-cache-dir flash-attn==2.5.5 --no-build-isolation

# --- OpenVLA -----------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/openvla/openvla.git
WORKDIR /opt/openvla
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir transformers==4.40.1

# --- LIBERO ------------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
WORKDIR /opt/LIBERO
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -r /opt/openvla/experiments/robot/libero/libero_requirements.txt

# transformers gets clobbered by transitive deps above. Re-pin last. This line is load-bearing.
RUN pip install --no-cache-dir transformers==4.40.1

ENV PYTHONPATH=/opt/LIBERO:${PYTHONPATH}

WORKDIR /opt/openvla
CMD ["/bin/bash"]
