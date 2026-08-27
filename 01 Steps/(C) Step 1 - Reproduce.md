---
type: project-step
project: VLA Policy Eval Layer
step: 1
status: in-progress
days: 1-2
tags:
  - eval-layer
---

# Step 1 — Reproduce the Baseline

**This is a hard gate. Nothing in Steps 2-6 begins until this passes.**

The single check the Qwen project never had: prove the pipeline is correct *before*
building anything on top of it.

---

## Done means

> OpenVLA-7B runs on unmodified LIBERO-Spatial and reproduces the published success rate
> within noise.

Concretely: run the official checkpoint on the standard suite, get a number, and compare it
to the number in the OpenVLA paper. If they disagree beyond the confidence interval, the
harness is wrong — **not** the policy. Fix the harness.

---

## Why a reproduction gate at all

Without it there is no way to tell these two apart:

| Observation | Interpretation A | Interpretation B |
|---|---|---|
| Success rate is 40% | The policy is weak under my conditions | My action decoding / observation preprocessing is broken |

For an *evaluation* project, being unable to distinguish these is fatal — harness
correctness is the entire product. The published baseline is the only external truth signal
available. Use it before it stops being useful.

---

## Order of operations

- [x] **1.1** Verify live, do not assume: current OpenVLA LIBERO checkpoint names, the LIBERO
      install path and API, `g6.xlarge` spot price and regional availability
- [x] **1.2** Stand up the EC2 GPU box + Docker image. MuJoCo headless rendering is the usual
      first fight — solve it here, not later — **DONE 2026-08-26**, see 1.2 outcome below
- [ ] **1.3** Run **one** episode of **one** task. Confirm the loop closes end to end:
      observation in -> action out -> sim steps -> episode terminates with a verdict
- [ ] **1.4** Log per-step observations and actions for that single episode. Eyeball them.
      Wrong action scaling and wrong image preprocessing both look like "policy is bad"
- [ ] **1.5** Run the full standard eval on LIBERO-Spatial. Record the per-task success rate
- [ ] **1.6** Compare against published. Record the baseline table in this file
- [ ] **1.7** Rank tasks by baseline success -> this selects the 5 tasks for the sweep (D-007)
- [ ] **1.8** Write the Step 1 decision entry in [[00 Design/(C) Decision Log.md]]

---

## 1.1 — Live verification (2026-08-20)

Verified against the OpenVLA GitHub README and Hugging Face, not memory.

### Checkpoints — confirmed live

| Suite | Checkpoint | Published SR (A100) |
|---|---|---|
| LIBERO-Spatial | `openvla/openvla-7b-finetuned-libero-spatial` | **84.7 ± 0.9%** |
| LIBERO-Object | `openvla/openvla-7b-finetuned-libero-object` | 88.4 ± 0.8% |
| LIBERO-Goal | `openvla/openvla-7b-finetuned-libero-goal` | 79.2 ± 1.0% |
| LIBERO-Long | `openvla/openvla-7b-finetuned-libero-10` | 53.7 ± 1.3% |
| — | Average | 76.5 ± 0.6% |

Fine-tuned via LoRA (r=32). A second family exists: `moojink/openvla-7b-oft-*` (OpenVLA-OFT
— parallel decoding + action chunking, higher SR, far faster inference). **Open decision D-012.**

### Reference eval command

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

- `--center_crop True` is **required** — training used random-crop augmentation (90% area).
  Omitting it is the #1 silent reproduction failure.
- Default run = 500 trials (10 tasks x 50 episodes). Override with `--num_trials_per_task`.

### Environment — pin exactly

Published numbers were obtained on: **Python 3.10.13 · PyTorch 2.2.0 · transformers 4.40.1 ·
flash-attn 2.5.5**, on an **NVIDIA A100**.

`transformers==4.40.1` is the hard pin — OpenVLA ships custom modeling code that breaks on
later versions. This dictates the Docker base image.

Install: clone https://github.com/Lifelong-Robot-Learning/LIBERO, `pip install -e .`, then
`pip install -r experiments/robot/libero/libero_requirements.txt`.

### Hardware findings — `g6.xlarge`

| Spec | Value | Consequence |
|---|---|---|
| GPU | 1x L4, 24 GB (Ada, sm_89) | 7B bf16 (~15 GB) fits. flash-attn 2 supports Ada. |
| vCPU | **4** | MuJoCo stepping is CPU-bound — likely the real bottleneck, not the GPU |
| System RAM | **16 GB** | Tight for staging a ~15 GB checkpoint into VRAM. Watch for OOM-kill on load. |
| On-demand | ~$0.80/hr us-east-1 | |
| Spot | ~$0.23/hr equivalent | 37 GPU-hr sweep ~ $9 spot / $30 on-demand |

**README caveat that matters to the gate:** *"Results may vary across GPU types due to
nondeterminism."* The published numbers are A100. Reproducing on L4 introduces a variance
source the paper did not have.


---

## 1.2 — Outcome (2026-08-26)

**Done means:** `verify_env.py` passes 8/8 from a **cold `docker build`** with zero manual steps.
Achieved on `g6.xlarge` on-demand (L4, sm_89, 23.7 GB).

```
[PASS] torch + CUDA: NVIDIA L4, 23.7 GB, torch 2.2.0+cu121, sm_89
[PASS] transformers pin: 4.40.1
[PASS] flash-attn: 2.5.5
[PASS] host RAM: 16.1 GB   <-- TIGHT for a 15 GB checkpoint load
[PASS] disk space
[PASS] MuJoCo headless (EGL): mujoco 3.12.0, rendered (128,128,3) offscreen via EGL
[PASS] LIBERO benchmark: libero_spatial loaded, 10 tasks,
       task[0]='pick up the black bowl between the plate and the ramekin and place it on the plate'
```

### The predicted fight was not the real fight

The plan said *"MuJoCo headless rendering is the usual first fight — solve it here, not later."*
**EGL rendered offscreen on the first attempt.** Every hour actually spent went to packaging.

### Four environment bugs, all now baked into the Dockerfile

1. **`cmake` missing.** `robosuite` pulls `egl_probe`, which builds native code. pip built every
   other wheel first, then died on it — and because pip installs atomically, *nothing* was
   installed. That accidental abort is the only reason bug 2 didn't land.
2. **Wrong requirements file.** LIBERO's own `requirements.txt` pins `transformers==4.21.1` and
   `numpy==1.22.4`. Installing it would have silently downgraded transformers 19 minor versions
   and broken OpenVLA's custom modeling code, plus broken numba (needs numpy >= 1.24).
   **The correct file is OpenVLA's `experiments/robot/libero/libero_requirements.txt`.**
3. **NumPy 2.x.** torch 2.2.0 is compiled against NumPy 1.x; under 2.x the bridge dies with
   `_ARRAY_API not found`, so `torch.from_numpy()` and `.numpy()` stop working — which would
   break every render→policy handoff. Pinned to 1.26.4 **after** the requirements install.
4. **LIBERO's editable install registers nothing.** `setup.py` uses `find_packages()`, which
   needs `__init__.py` at every level. `/opt/LIBERO/libero/` has none — the real package is the
   *inner* `/opt/LIBERO/libero/libero/`. So `pip show libero` reports success while
   `import libero` raises `ModuleNotFoundError`. It is an implicit namespace package; the fix is
   `ENV PYTHONPATH=/opt/LIBERO`, and the correct import is the double
   `from libero.libero import benchmark`.

Also baked: `~/.libero/config.yaml`, because `libero.libero` prompts **interactively** for dataset
paths on first import. In a container that reads as a hang — and inside an unattended Step 3 sweep
it would be a hang at 3am with a GPU meter running.

### Two of the bugs were in the gate script, not the environment

> [!important] The check being wrong costs more than having no check
> - **`try/except` swallowed an interactive prompt.** LIBERO's config prompt surfaced as a
>   one-line failure instead of a question, and sent debugging in the wrong direction for hours.
> - **The disk check measured the wrong filesystem.** It read `shutil.disk_usage("/")` — the EBS
>   root, 37 GB free — and failed. The checkpoint actually lands in `HF_HOME`, bind-mounted to the
>   instance-store NVMe with **217 GB free**. A false FAIL on a healthy environment.
>
> Fixed: `disk()` now resolves `HF_HOME`. Both are worth remembering — for an eval project whose
> whole product is a trustworthy harness, **a false negative in the validation layer is the most
> expensive kind of bug.**

### Environment-variable scoping — the one that cost the most

`export PYTHONPATH=...` does not survive a new shell or a container restart. The fix kept being
applied in a different shell than the one running the script, so each verification reproduced the
original error and looked like the fix had failed. Now `ENV` in the Dockerfile.

### Recorded for the baseline table

**GPU: NVIDIA L4 (sm_89), not A100.** The published numbers are A100 and the OpenVLA README warns
results vary across GPU types. Every number this project produces is measured against an L4
baseline; that must stay stated wherever the baseline is quoted.

## Baseline results

_Fill this in. It becomes the reference every later number is measured against._

| Task | Published SR | My SR | Trials | Delta | In noise? |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

**Selected 5 tasks for the sweep:**

1.
2.
3.
4.
5.

---

## Failure notes

_When reproduction disagrees, log what was wrong and how it was found. This is the most
transferable knowledge the whole project will produce — every eval harness hits these._

| Symptom | Root cause | How found |
|---|---|---|
|  |  |  |

Usual suspects, in rough order of likelihood: action un-normalisation mismatch, image
resize/crop differing from training, wrong camera view fed to the model, off-by-one in the
action chunk, episode step limit set differently from the paper, seed handling.

---

## Out of scope for Step 1

No paraphrase. No quantization. No S3. No report. One policy, one suite, one number.
