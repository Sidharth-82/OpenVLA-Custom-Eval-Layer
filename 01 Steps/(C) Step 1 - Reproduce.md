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
- [ ] **1.2** Stand up the EC2 GPU box + Docker image. MuJoCo headless rendering is the usual
      first fight — solve it here, not later
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
