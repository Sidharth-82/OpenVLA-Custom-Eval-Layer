# Step 1.2 — Environment

Build and run the reproduction container. Not a note; lives with the code, so no `(C)` prefix.

## Build

Build on the **laptop**, not on `g6.xlarge`. The instance has 4 vCPU — a flash-attn source
build there costs 30-60+ min of paid GPU time for a pure-CPU task. Docker Desktop can build
a CUDA image without a local GPU.

```bash
docker build -t vla-eval:repro -f src/docker/Dockerfile .
```

Then push to ECR and pull on the instance, or `docker save | ssh ... docker load` for a
one-off.

## Run

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $HOME/.cache/huggingface:/cache/hf \
  -v $PWD/results:/opt/openvla/results \
  vla-eval:repro
```

- `--gpus all` — without it `torch.cuda.is_available()` is False and every check fails.
- `--shm-size=8g` — the 64 MB default causes opaque DataLoader worker crashes.
- Mounting the HF cache means a killed spot instance doesn't re-download ~15 GB.

## Verify before running anything

```bash
python /opt/openvla/../scripts/verify_env.py   # or mount src/scripts and run directly
```

Seven checks, each mapped to a known silent failure. **A clean pass is the definition of
1.2 being done.** Do not download a checkpoint before this passes.

## 1.3 — one episode, one task

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1
```

`--center_crop True` is **not optional** — training used random-crop augmentation at 90%
area. Omitting it makes a working policy look broken, which is precisely the
Interpretation A vs B confusion the reproduction gate exists to prevent.

## Known traps

| Trap | Symptom | Fix |
|---|---|---|
| `MUJOCO_GL` unset | `GLFWError: X11 display not found` | Already `egl` in the image; don't override |
| transformers upgraded by a transitive dep | `KeyError` on the OpenVLA config / unknown model type | Re-pin `4.40.1`, last line of the Dockerfile |
| Host RAM exhaustion on load | Container killed, exit 137, no traceback | 16 GB is tight; use `low_cpu_mem_usage=True`, or size up |
| `--shm-size` default | DataLoader workers die silently | `--shm-size=8g` |
| Spot reclaim mid-eval | Partial results, no record | Checkpoint results to disk per episode, sync to S3 (Step 2) |
