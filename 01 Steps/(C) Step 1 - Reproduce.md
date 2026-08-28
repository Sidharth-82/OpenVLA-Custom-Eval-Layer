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
- [x] **1.3** Run **one** episode of **one** task. Confirm the loop closes end to end:
      observation in -> action out -> sim steps -> episode terminates with a verdict
      — **DONE 2026-08-27**, see 1.3 outcome below
- [x] **1.4** Log per-step observations and actions for that single episode. Eyeball them.
      Wrong action scaling and wrong image preprocessing both look like "policy is bad"
      — **DISCHARGED BY EVIDENCE 2026-08-27**, see 1.4 outcome below
- [x] **1.5** Run the full standard eval on LIBERO-Spatial. Record the per-task success rate
      — **DONE 2026-08-27. 418/500 = 83.6%. Gate PASSED.**
- [x] **1.6** Compare against published. Record the baseline table in this file
      — **DONE 2026-08-27, see below**
- [x] **1.7** Rank tasks by baseline success -> this selects the 5 tasks for the sweep (D-007)
      — **DONE 2026-08-27, see below**
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


---

## 1.3 — Outcome (2026-08-27)

**Done means:** an episode terminates with a verdict. Achieved, and then some — the full
LIBERO-Spatial suite ran at one trial per task.

**Hardware:** `g5.2xlarge` on-demand — **NVIDIA A10G (sm_86), 24 GB VRAM, 8 vCPU, 32 GB RAM.**
Note this differs from 1.2, which ran on a `g6.xlarge` (L4, sm_89); g6 capacity was unavailable.
The extra host RAM cleared the "TIGHT for a 15 GB checkpoint" warning outright.

### Command

```bash
python experiments/robot/libero/run_libero_eval.py   --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial   --task_suite_name libero_spatial   --center_crop True   --num_trials_per_task 1   --seed 7   --local_log_dir /workspace/results/step1_3
```

`--local_log_dir` is overridden deliberately: the default `./experiments/logs` sits inside
`/opt/openvla`, which is not a mounted volume and dies with the container.

**OpenVLA's script was used, not a custom runner — deliberately.** Step 1 is the truth anchor
(D-003). A custom loop here would make "the harness is wrong" and "the environment is wrong"
indistinguishable. The custom runner is Step 2, and it earns the right to exist by matching this.

### Four more pins were required

1.2 established the environment builds; it did not establish the eval *runs*. Three further
version conflicts surfaced, each in a package **left unpinned upstream**:

| Package | Resolved to | Broke because | Pinned to |
|---|---|---|---|
| `tensorflow-metadata` | 1.21.0 | generated pb2 needs `protobuf>=5.27`; TF 2.15 caps protobuf `<5` -> `ImportError: cannot import name 'runtime_version'` | **1.14.0** |
| `wandb` | 0.28.0 | ships pb2 built for protobuf 5.x -> `ImportError: cannot import name 'Imports' from wandb_telemetry_pb2` | **0.17.9** |
| `mujoco` | 3.12.0 | `robosuite==1.4.0` targets the MuJoCo 2.3.x bindings; 3.x changed them -> `AssertionError` in `get_joint_qpos_addr` | **2.3.7** |

Together with `numpy` (1.2) that is **four transitive dependencies**, none pinned by LIBERO or
OpenVLA, each of which resolved forward to a version incompatible with the pinned packages
around it.

> [!important] This is the finding, not the annoyance
> **The published baseline's own repository no longer installs cleanly.** Reproducing a number
> from the literature required pinning four packages the authors never pinned. Nothing was wrong
> with their code — the environment underneath it drifted.
>
> That is the reproducibility problem in miniature, encountered first-hand rather than read about,
> and it is the argument for why an evaluation harness should own a **thinner dependency surface
> than the training repo it evaluates**. Every one of those four came in through
> `prismatic/__init__.py` eagerly importing the RLDS *training* data loader — a code path this
> project never executes (D-002).

### Results — 8 / 10, single trial per task

| # | Task | Verdict | Wall time |
|---|---|---|---|
| 1 | black bowl between the plate and the ramekin | ✅ | 51 s |
| 2 | black bowl next to the ramekin | ✅ | 47 s |
| 3 | **black bowl from table center** | ❌ | **90 s** |
| 4 | black bowl on the cookie box | ✅ | 37 s |
| 5 | black bowl in the top drawer of the wooden cabinet | ✅ | 53 s |
| 6 | black bowl on the ramekin | ✅ | 52 s |
| 7 | black bowl next to the cookie box | ✅ | 46 s |
| 8 | **black bowl on the stove** | ❌ | **91 s** |
| 9 | black bowl next to the plate | ✅ | 57 s |
| 10 | black bowl on the wooden cabinet | ✅ | 49 s |

**80.0%** against a published **84.7 ± 0.9%**. Total run 10:05. Rollout MP4s written per episode.

> [!danger] This is NOT a reproduction. Do not describe it as one.
> **n = 10. Wilson 95% CI = [49.0%, 94.3%] — 45.3 points wide.**
>
> The interval contains the published 84.7%. It also contains 50% and 94%. This run **cannot
> distinguish a correct harness from a badly broken one**, and the 4.7pp gap to the published
> number carries no information whatsoever.
>
> This is exactly the trap D-013 was written against. The recorded bad reasoning was
> *"5pp keeps the score between 80-90%, which is a good eval result."* A 4.7pp gap at n=10 is the
> same error in better clothing. **The gate is tested at n=500, in 1.5.**
>
> | n | Result | Wilson 95% CI | Width |
> |---|---|---|---|
> | 10 | 8/10 = 80.0% | [49.0, 94.3] | **45.3 pp** |
> | 50 | 42/50 = 84.0% | [71.5, 91.7] | 20.2 pp |
> | 500 | 423/500 = 84.6% | [81.2, 87.5] | **6.3 pp** |
>
> That table is D-004 and D-013 as a picture, and it is why the ladder is ordered the way it is.

### Free observation: the failures are timeouts

Both failures ran **~90 s**; every success finished in **36-57 s**. The failed episodes exhausted
the step budget rather than grabbing the wrong object and stopping.

On a 10-episode smoke run, binary success says only *"two failed."* Duration already says *how*.
That is **D-011's** dense-progress argument, observed rather than asserted — and it is the first
piece of evidence that the failure taxonomy in Step 4 has something real to classify.

### Noise, catalogued so it is not re-debugged

- `Unable to register cuDNN/cuFFT/cuBLAS factory` — TF and PyTorch both loading CUDA libs
- `NUMA node read from SysFS had negative value (-1)` — container doesn't expose NUMA topology
- `TF-TRT Warning: Could not find TensorRT`, `robosuite: No private macro file`, gym deprecation
- **`Exception ignored in: MjRenderContext.__del__` / `EGLError: EGL_NOT_INITIALIZED`** — EGL
  teardown ordering at interpreter shutdown, *after* all results were logged. "Exception ignored"
  means Python swallowed it. Would only matter if contexts leaked mid-sweep

### Carried into 1.4

`WARNING: No local dataset_statistics.json file found for current checkpoint ... you may run into
errors ... due to an absent unnorm_key.` Episodes succeeded, so un-normalisation is evidently
working — but 1.4 exists to confirm that by inspection rather than by inference.


---

## 1.4 — Outcome (2026-08-27): discharged by evidence, not by inspection

**1.4 was written to catch a failure that did not happen.** Its purpose was to detect silent
harness bugs — wrong action un-normalisation, wrong image preprocessing — that present as
*"the policy is bad"* with no error.

**1.3 returned 8/10.** Neither of those bugs is compatible with that result: a broken
`unnorm_key` or an omitted center-crop produces near-zero success, not eight successful
pick-and-places. The `dataset_statistics.json` warning at load time is therefore benign.

Recorded rather than deleted, because *"the check was designed for a failure mode the data
ruled out"* is a legitimate outcome and a different thing from skipping a step.

**Per-step action/observation logging is not abandoned — it is reassigned.** Its remaining value
is as **Step 4 groundwork** (the failure taxonomy cannot be built without per-step data), which
makes it the runner's job in Step 2, not a debugging step in Step 1.

### Substituted: visual inspection of the rollout MP4s

The GPU was occupied by the 1.5 reproduction run, so episodes could not be re-run instrumented.
The rollout videos were watched instead.

| Episode | Object | Approach | Grasp | Place | Notes |
|---|---|---|---|---|---|
| **3** — bowl from table center | ✅ correct | ✅ correct | ❌ **failed** | — | No drift |
| **8** — bowl on the stove | ✅ correct | ✅ correct | ✅ | ❌ **failed** | **Heavy jitter** near the target |

### The observation: both failures are downstream of perception

The policy identified the correct object and brought the arm to it in **both** failures. What
broke was the **contact-rich phase** — closing the grasp, and completing the place. Neither
failure was a perception error, a wrong-object error, or a language-grounding error.

> [!warning] n = 2. This is a hypothesis, not a finding.
> Two episodes is an anecdote. The 500-episode run and the Step 4 taxonomy are what would test it.
> State it that way out loud; do not let it harden into a claim.

### Consequence: a live threat to D-005, to be raised rather than discovered

**D-005 selects language paraphrase as the v1 perturbation axis**, reasoning that it probes the
language half of a VLA and separates genuine grounding from scene pattern-matching.

If fragility is concentrated in the **manipulation** phase rather than the **grounding** phase,
paraphrase may move the success rate very little — not because language is unimportant, but
because the policy fails at contact regardless of how the instruction is worded. **The axis would
then be measuring a small effect layered on top of a failure mode it does not touch.**

This does not invalidate D-005. It remains the highest insight-per-engineering-hour axis, and a
null result is itself informative: *"language paraphrase does not degrade success; failure is
dominated by manipulation"* is a real finding, and arguably a more interesting one than a
degradation curve. But it is a threat to validity, and it belongs in the report **stated up front**,
in the same way D-007's biased task selection is.

### The jitter is evidence for D-011

Decoding is deterministic (D-010), so the jitter in episode 8 is not sampling noise — it is the
policy oscillating between action tokens as the observation shifts near the goal.

**That is invisible in a binary success flag and obvious in a per-step action log.** It is the
most concrete argument the project has produced for recording a dense progress signal, and it is
a direct answer to *"why not just log success?"*


---

## 1.5 / 1.6 — Reproduction (2026-08-27)

> [!success] The gate passed
> ```
> RESULT          418 / 500  =  83.6%
> PUBLISHED       84.7% +/- 0.9   (n = 1500, A100)
> GAP             1.1 pp
> GATE (D-013)    +/- 5.0 pp      ->  PASS
>
> Wilson 95% CI   [80.1%, 86.6%]   width 6.5 pp
> Published 84.7% falls INSIDE the interval.
> ```

**Run configuration.** `run_libero_eval.py`, `--task_suite_name libero_spatial`,
`--num_trials_per_task 50`, `--center_crop True`, `--seed 7`,
checkpoint `openvla/openvla-7b-finetuned-libero-spatial`, bf16 + flash-attention.
**Hardware: `g5.2xlarge`, NVIDIA A10G (sm_86), 24 GB VRAM, 32 GB host RAM.**
Wall clock ~8.5 h. Log: `results/step1_5/`.

### The derivation validated itself

D-013 estimated the standard error at n=500 as **~1.6 pp**, implying a 95% interval
**~6.3 pp** wide. The observed interval is **6.5 pp** wide.

**The pre-run statistical model of the experiment was accurate to 0.2 percentage points.** That
is a separate and stronger result than the pass: it says the reasoning behind the gate was sound
regardless of where the number happened to land. Had the result fallen outside ±5pp, this would
still hold — and the gate would have been failed honestly.

### How this may and may not be described

> [!warning] Precision matters here more than anywhere else in the project
> **Correct:** *"The result is consistent with the published baseline. 83.6% against 84.7%, inside
> the ±5pp gate I fixed before the run, and the published value falls within my 95% interval."*
>
> **Not correct:** *"I matched the paper."* 83.6 ≠ 84.7. What has been established is that the
> difference is not distinguishable from sampling and hardware variance at this sample size —
> which is a failure to reject, not a demonstration of equality.
>
> **Also not correct:** treating the 1.1 pp gap as evidence the gate should have been ±2. The gate
> was set from the sampling maths before the run and is not revised by its own outcome. Narrowing
> it retroactively is the same goalpost-moving error in the opposite direction.

### What the gate now licenses

Step 1 was a hard gate: nothing downstream begins until reproduction passes. It has passed, so
the harness is now a **trusted instrument**. Any degradation measured in Steps 3-5 can be
attributed to the perturbation rather than to the harness — which is the entire reason this step
exists and the check the Qwen project never had (D-001).

### Carried forward

- **1.7 — per-task ranking.** The final task (`black bowl on the wooden cabinet`) closed at
  **0.74**, well under the 0.836 aggregate, so per-task spread is real and the five-task selection
  (D-007) will be a meaningful cut rather than an arbitrary one. Extract all ten rates from the log.
- **Baseline is A10G, not A100.** Every number this project produces from here is measured against
  *this* run on *this* silicon. Quote the hardware whenever the baseline is quoted.
- **500 rollout MP4s** were written. Retrieve before terminating the box; they are the raw material
  for the Step 4 failure taxonomy.

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


---

## 1.7 — Per-task baseline and sweep selection (2026-08-27)

Per-task rates mean to **0.836**, exactly matching the aggregate — a clean consistency check on
the log parse.

| idx | rate | Wilson 95% (n=50) | width | task |
|---|---|---|---|---|
| **6** | **0.90** | [78.6, 95.7] | 17.0 | next to the cookie box |
| **3** | **0.90** | [78.6, 95.7] | 17.0 | on the cookie box |
| **1** | **0.88** | [76.2, 94.4] | 18.2 | next to the ramekin |
| **0** | **0.88** | [76.2, 94.4] | 18.2 | between the plate and the ramekin |
| **7** | **0.86** | [73.8, 93.0] | 19.2 | on the stove |
| 8 | 0.82 | [69.2, 90.2] | 21.0 | next to the plate |
| 4 | 0.80 | [67.0, 88.8] | 21.8 | in the top drawer of the wooden cabinet |
| 2 | 0.80 | [67.0, 88.8] | 21.8 | from table center |
| 5 | 0.78 | [64.8, 87.2] | 22.5 | on the ramekin |
| 9 | 0.74 | [60.4, 84.1] | 23.7 | on the wooden cabinet |

**Sweep selection (D-007): task indices `[0, 1, 3, 6, 7]`.** Observed range 0.74–0.90, a 16 pp
spread — so the cut is a real one rather than an arbitrary slice of a flat distribution.

### Second threat to validity — the ranking itself is noisy

> [!warning] Say this before it is asked
> **At n=50 per task, every interval overlaps every other interval.** The 5th/6th boundary is
> 0.86 `[73.8, 93.0]` against 0.82 `[69.2, 90.2]` — near-total overlap. **The top-5 cut is not
> statistically defensible as "these are the five easiest tasks."**
>
> It does not invalidate the selection. D-007's purpose is **measurement headroom in aggregate**,
> and a 0.86–0.90 band beats a 0.74–0.80 band for that regardless of whether the within-band
> ordering is real. The selection is a pragmatic choice about where degradation is measurable,
> not a claim about true task difficulty.
>
> **But it is a second threat to validity, on top of the biased-sample one already recorded in
> D-007, and both belong in the report.** Selecting on a noisy estimate of the outcome variable
> also invites regression to the mean: tasks chosen because they scored high may score lower on
> a re-run for no reason other than sampling.

**Consequence for the sweep budget:** per-cell n=50 gives ~±10 pp per task. Aggregating the five
selected tasks into a single per-level success rate (n=250 per paraphrase level) is what makes
the degradation measurement tight enough to act on. **Per-task degradation curves at n=50 will
not be individually conclusive** — that is a reporting decision to make before Step 5, not after.

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
