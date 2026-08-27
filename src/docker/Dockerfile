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
# cmake: robosuite pulls egl_probe, which builds native code with CMake. Without it the
#        libero_requirements install below dies at "RuntimeError: CMake must be installed"
#        AFTER building every other wheel. Verified 2026-08-26.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential cmake \
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

# NOTE: there is deliberately no `pip install -e .` here.
# LIBERO's setup.py uses find_packages(), which requires __init__.py at every level.
# /opt/LIBERO/libero/ has none — the package with __init__.py is the *inner*
# /opt/LIBERO/libero/libero/. So find_packages() never descends, the editable install
# registers an EMPTY mapping, and `pip show libero` reports success while
# `import libero` raises ModuleNotFoundError. Diagnosed 2026-08-26; cost most of an evening.
# It is an implicit namespace package: putting the repo root on PYTHONPATH is the fix.
# Correct import is the double: `from libero.libero import benchmark`.

# Use OPENVLA's libero_requirements.txt, NOT LIBERO's own requirements.txt.
# LIBERO's pins transformers==4.21.1 and numpy==1.22.4, which would silently break
# OpenVLA's custom modeling code and numba. OpenVLA's file is curated to avoid that.
RUN pip install --no-cache-dir -r /opt/openvla/experiments/robot/libero/libero_requirements.txt

# --- Re-pin last. These two lines are load-bearing. --------------------------
# Transitive deps above drag both of these off their required versions.
# numpy MUST stay <2: torch 2.2.0 is compiled against NumPy 1.x, and under NumPy 2.x
# the torch<->numpy bridge dies with "_ARRAY_API not found" — torch.from_numpy() and
# .numpy() stop working, which breaks every render->policy handoff in the harness.
RUN pip install --no-cache-dir transformers==4.40.1 numpy==1.26.4

# --- protobuf triangle -------------------------------------------------------
# Three packages fight over protobuf, and two of them are unpinned upstream, so a
# fresh build resolves them forward and breaks. Diagnosed 2026-08-27:
#
#   tensorflow 2.15.0    -> requires protobuf <5
#   tensorflow-metadata  -> unpinned; latest (1.21) needs protobuf >=5.27
#                           => ImportError: cannot import name 'runtime_version'
#   wandb                -> unpinned; latest (0.28) ships pb2 built for protobuf 5.x
#                           => ImportError: cannot import name 'Imports' from wandb_telemetry_pb2
#
# protobuf lands at 4.25.x to satisfy TensorFlow, so the other two must come back to it.
#
# NOTE: none of this stack is used by this project. It is dragged in by
# prismatic/__init__.py eagerly importing the RLDS *training* data loader
# (tensorflow -> tensorflow_datasets -> dlimp), and this project never trains (D-002).
# Pinned here because the published baseline's own repo no longer installs cleanly
# without it — which is itself the reproducibility problem this harness exists to catch.
RUN pip install --no-cache-dir "tensorflow-metadata==1.14.0" "wandb==0.17.9"

# --- mujoco 2.3.x, NOT 3.x ---------------------------------------------------
# robosuite==1.4.0 (pinned by libero_requirements) targets the MuJoCo 2.3.x bindings.
# mujoco is unpinned, so a fresh build resolves it to 3.x, whose binding internals changed:
#
#   robosuite/utils/binding_utils.py:521, get_joint_qpos_addr()
#   assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
#   => bare AssertionError the moment LIBERO instantiates an env.
#
# 2.3.7 is the last of the 2.3 line and still provides mujoco.Renderer, which
# src/scripts/verify_env.py uses for the EGL headless check. Diagnosed 2026-08-27.
RUN pip install --no-cache-dir "mujoco==2.3.7"

# --- LIBERO runtime config ---------------------------------------------------
# On first import, libero.libero prompts INTERACTIVELY for dataset paths and writes
# ~/.libero/config.yaml. In a container that prompt reads as a hang, and inside a
# try/except it surfaces as an unrelated-looking failure. Bake it so it never asks.
# `datasets` is intentionally an empty dir: this project is inference-only (D-002) and
# never touches the ~100 GB LIBERO demonstration set. bddl_files and init_files ship
# in the repo and are the only assets an eval run actually needs.
RUN mkdir -p /root/.libero /datasets && \
    printf '%s\n' \
      'assets: /opt/LIBERO/libero/libero/./assets' \
      'bddl_files: /opt/LIBERO/libero/libero/./bddl_files' \
      'benchmark_root: /opt/LIBERO/libero/libero' \
      'datasets: /datasets' \
      'init_states: /opt/LIBERO/libero/libero/./init_files' \
      > /root/.libero/config.yaml

ENV PYTHONPATH=/opt/LIBERO:${PYTHONPATH}

WORKDIR /opt/openvla
CMD ["/bin/bash"]
