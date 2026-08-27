# Design Decisions

An evaluation layer for vision-language-action policies. Not a model, not a training run —
a harness that answers **when does this policy break, and is the difference real?**

This file is the project's reasoning, written as the decisions were made rather than
reconstructed afterwards. For an eval project the reasoning *is* the product: a number
nobody can interrogate is not a result.

**Status:** design complete, implementation in progress. Step 1 of 6.

---

## Scope

| Piece | Choice |
|---|---|
| Task suite | LIBERO (MuJoCo / robosuite), LIBERO-Spatial |
| Policy A | OpenVLA-7B, official LIBERO-finetuned checkpoint, bf16 |
| Policy B | OpenVLA-7B, 4-bit quantized |
| Perturbation axis | Language instruction paraphrase (v1) |
| Trials | 50 per cell, 50 distinct initial states, deterministic decoding |
| Execution | EC2 L4 GPU, Docker, results to S3 |

**No training happens in this project.** If a plan involves a training run, it is out of scope.

---

## D-001 · Build an eval layer, not repair an unexecuted training pipeline

- **Chose:** start an evaluation project.
- **Over:** debugging a half-written VLM fine-tuning script into a working state.
- **Why:** the script had never been executed end to end — no confirmed-good state to return
  to. Repairing it means paying for GPU training time to validate a pipeline with no known
  baseline. Sunk cost is real; forward cost was worse.
- **General principle:** a project whose failure mode is *silent* and whose iteration cost is
  *high* is a bad bet under limited compute. Eval inverts both — failures are loud and
  iteration is cheap.

## D-002 · Evaluation over training

- **Chose:** build a harness around an off-the-shelf policy.
- **Over:** training or fine-tuning a policy.
- **Why:** eval is inference-only. There is no multi-day training run to get silently wrong,
  failures surface in minutes rather than days, and the failure output *is* the deliverable
  rather than a diagnostic on the way to one.
- **Cost accepted:** produces no training track record. Stated plainly rather than blurred.

## D-003 · LIBERO over ManiSkill3

- **Chose:** LIBERO (MuJoCo / robosuite).
- **Over:** ManiSkill3 (SAPIEN, GPU-parallel envs), RoboCasa.
- **Why:** LIBERO has **published baselines from multiple VLA papers.** That provides a truth
  anchor: if the harness reproduces the paper's number, the harness is correct. Without one,
  "the policy is bad" and "my harness is broken" are indistinguishable — and for an eval
  project, harness correctness is the entire product.
- **Cost accepted:** ManiSkill3's GPU-parallel envs offer far higher episode throughput.
  Traded throughput for verifiability; throughput was not the binding constraint.

## D-004 · Depth over breadth in the sweep

- **Chose:** 50 trials per cell on a single perturbation axis.
- **Over:** ~15 trials per cell across 8 axes.
- **Why:** at 50% success, 50 binomial trials give roughly a ±14% Wilson interval; 15 trials
  give roughly ±25%. At 15 trials a genuine 10-point drop is indistinguishable from noise.
  The wide sweep produces a colourful chart of undefendable conclusions.
- **General principle:** decide the smallest effect worth detecting *before* choosing the
  sample size, not after.

## D-005 · Language paraphrase as the v1 axis

- **Chose:** instruction paraphrase as the only v1 perturbation axis.
- **Over:** object pose jitter, camera extrinsics jitter, distractor objects.
- **Why:** highest insight per hour of engineering — pure string manipulation, zero simulator
  work — and it targets the *language* half of a vision-language-action model, which the
  standard benchmark barely probes. It separates genuine language grounding from scene
  pattern-matching.
- **Cost accepted:** camera-extrinsics jitter maps more directly to real deployment. Parked
  as the strongest v1.5 candidate, not rejected.

## D-006 · 4-bit quantization as policy version B

- **Chose:** OpenVLA-7B 4-bit quantized as the second version for regression tracking.
- **Over:** Octo-small (different architecture), degraded-observation variants.
- **Why:** same weights, one flag — no second toolchain inside a fixed time box. And it asks
  a question with deployment stakes: *does on-robot quantization cost accuracy, and is the
  cost uniform or concentrated in specific slices?*
- **Cost accepted:** Octo-small (27M, diffusion head, JAX) would give a richer architectural
  comparison. Rejected on integration risk, not on scientific merit.
- **Under revision:** see D-012 — OpenVLA-OFT is now the stronger Policy B candidate.

## D-007 · Five highest-baseline tasks, not all ten

- **Chose:** restrict the sweep to the 5 LIBERO-Spatial tasks with the highest baseline
  success rate.
- **Over:** all 10 tasks; or 5 chosen at random.
- **Why:** **measurement headroom.** A task at 20% baseline cannot demonstrate degradation —
  a floor effect masks the thing being measured. High-baseline tasks maximise detectable
  dynamic range per GPU-hour.
- **Threat to validity, stated in the report:** selecting on baseline success is a *biased
  sample* of the suite. The claim is "paraphrase degrades the tasks the policy is best at,"
  which is narrower than "paraphrase degrades LIBERO-Spatial." Not overstated.

## D-008 · Axes ship one at a time, in a fixed order

- **Chose:** ship axis 1 complete through all six steps, then add axis 2, then axis 3.
- **Over:** building two or three axes into v1 simultaneously.
- **Why:** each axis costs ~1 day of implementation plus 1–2 days of GPU sweep. More
  importantly, a full vertical slice through all six steps on one axis proves the layer works
  end to end. Two half-built axes prove nothing.
- **Cost accepted:** the v1 finding is narrower — one dimension of robustness, not a profile.
  A narrow finding that ships beats a broad one that doesn't.

## D-009 · Perturbations are pluggable transforms, not runner features

- **Chose:** the runner accepts a perturbation object that transforms the scenario spec. The
  runner knows nothing about paraphrasing, pose jitter, or distractors.
- **Over:** implementing the paraphrase axis directly inside the run loop.
- **Why:** direct consequence of D-008. If axis 1 is hardcoded, axis 2 is a rewrite and axis 3
  is a rewrite on top of a rewrite — which would cost more than building them together and
  defeat the point of sequencing. With a transform interface, a new axis is a module plus
  config.
- **General principle:** this is what separates an eval *layer* from an eval *script*. The
  layer is the part that does not change when the experiment changes.

## D-010 · A trial is a distinct initial scene state, with deterministic decoding

- **Chose:** 50 trials = 50 distinct LIBERO initial states, decoding deterministic.
- **Over:** (a) one initial state with 50 stochastic rollouts; (b) 50 states *and* stochastic
  decoding.
- **Why:** this determines **what the confidence interval actually measures.** Distinct initial
  states means the CI describes generalization across scene configurations. One state with
  stochastic rollouts means the CI describes decoder sampling randomness — a real but far
  narrower quantity, and not what anyone means by robustness. Option (b) confounds both
  sources and needs many more trials to separate them.
- **Consistency bonus:** this is also the published LIBERO protocol, which Step 1 must match
  to reproduce the baseline. One protocol for reproduction and perturbation means every
  perturbed cell is directly comparable to the baseline, with no bridging assumptions.
- **Watch for:** the same 50 init states must be reused across every level and every policy
  version — paired comparison, not independent samples.

## D-011 · Record a dense progress signal, not just binary success

- **Chose:** log binary success plus per-step state — minimum distance to target, grasp flag,
  lift flag.
- **Over:** binary success only (the published protocol); or binary plus time-to-completion.
- **Why:** binary success is too low-information to build a failure taxonomy on. A policy that
  reaches, grasps and drops at the last moment scores identically to one that never moved —
  and the taxonomy step exists precisely to tell those apart.
- **Cost accepted:** more storage per episode and some robosuite instrumentation. Binary
  success is still recorded unchanged, preserving comparability with published numbers.
- **General principle:** log intermediate state during the expensive run. A rerun to collect a
  metric that could have been recorded the first time costs full GPU price.

## D-012 · Original OpenVLA checkpoints as Policy A, not OpenVLA-OFT

- **Chose:** `openvla/openvla-7b-finetuned-libero-spatial` — LoRA r=32, bf16, autoregressive
  discretised action tokens.
- **Over:** `moojink/openvla-7b-oft-*` — parallel decoding + action chunking, ~97% SR, roughly
  an order of magnitude faster inference.
- **Why:**
  1. **Attribution.** Roughly seven conventions sit between observation and action: image
     crop/resize, prompt format, token-to-bin detokenisation, un-normalisation statistics,
     gripper sign, delta-vs-absolute action space, control frequency. Every one of them fails
     the *same way* — the robot moves badly — so they are indistinguishable from the outcome
     alone. OFT adds chunk size, chunk execution policy and chunk timing alignment: three
     more, and the least documented of the set. On the run whose entire purpose is finding my
     own bugs, fewer conventions wins.
  2. **The taxonomy needs failures.** Across the 1,000-episode sweep an 84.7% baseline yields
     ~150 failures to classify; a 97% baseline yields ~30. Spread across five categories that
     is six per bucket. **The baseline choice silently sets the sample size of the failure
     taxonomy, three steps before that step is reached.**
  3. **Headroom.** The v1 finding is degradation under paraphrase. 97% is pressed against the
     ceiling — the same rule that rejects a 20% task rejects a 97% policy. Wilson intervals
     are specified precisely because the normal approximation misbehaves near p = 1.
- **Cost accepted:** the sweep is substantially slower and more expensive than OFT would be.
  Traded throughput for attributability.
- **Parked:** OFT becomes the natural Policy B — autoregressive vs parallel decoding is a more
  interesting regression than 4-bit quantization of identical weights.

## D-013 · Reproduction gate: aggregate at ±5pp, per-task recorded as diagnostic

- **Chose:** `PASS if |aggregate_SR − 84.7| <= 5.0 pp` over the full 500-episode
  LIBERO-Spatial run. Per-task rates are recorded and inspected but do not decide the gate.
  **Threshold fixed before the run.**
- **Over:** aggregate at ±3pp; per-task matching on all 10 tasks, with or without a Bonferroni
  correction.
- **Baseline:** 84.7 ± 0.9% (OpenVLA paper). That ±0.9 implies n ≈ 1500 — 3 seeds × 500.
- **Where ±5 comes from:**

  ```
  SE_mine  (n = 500)  = sqrt(0.847 * 0.153 / 500) = 1.61 pp
  SE_paper (n = 1500) =                             0.93 pp

  errors independent  ->  add in quadrature:
  SE_diff  = sqrt(1.61^2 + 0.93^2)                = 1.86 pp
  95% band = 1.96 * 1.86                          = 3.64 pp

  round up for GPU-type variance (L4 vs A100)     -> 5 pp
  ```

  Two assumptions doing two different jobs: *independence* is what licenses adding in
  quadrature, *approximate normality* is what licenses the 1.96.
- **Why not per-task gating:** at n = 50 a single task's 95% interval is ±10pp — too wide to
  catch a real harness bug. And across 10 independent checks the chance of at least one false
  alarm is `1 − 0.95^10 = 40%`. A correct harness would look broken 4 runs in 10.
- **Why the threshold is pre-registered:** if the number is seen first, "within noise" quietly
  becomes whatever the result needs it to be. Pre-committing is what makes a pass mean
  something.

---

## The ladder

| Step | Done means |
|---|---|
| 1. Reproduce | OpenVLA on unmodified LIBERO matches published SR within the D-013 gate |
| 2. Runner | YAML scenario spec, seeded, containerized, JSONL to S3. Same seed twice = same result |
| 3. Paraphrase axis | 4 instruction levels version-controlled; sweep executes end to end |
| 4. Failure taxonomy | Every failed episode auto-classified: grasp miss / wrong object / collision / timeout / drift |
| 5. Slice + stats | Success rate per slice with Wilson CIs — "is this delta real?" answered honestly |
| 6. Regression + report | Two policy versions, one command, one HTML report |

**Step 1 is a hard gate.** Nothing else begins until reproduction passes.

## D-014 · Compute is disposable: minimal EBS, everything heavy on instance store

- **Chose:** keep the EBS root volume small and put the Docker data-root, image build, and
  Hugging Face cache on the instance-store NVMe. Terminate the box at the end of each step.
- **Over:** provisioning a large EBS root and treating the instance as persistent.
- **Why:** every artifact on that box is **regenerable** — the image rebuilds from the Dockerfile,
  the checkpoint re-downloads from Hugging Face, and results are published to S3 before the box is
  released. Nothing unique lives there, so paying for durable storage buys nothing. The instance
  store is included in the hourly price, is substantially faster than gp3 for the layer writes a
  docker build is dominated by, and on this instance offers 217 GB against a nearly full 145 GB
  root.
- **Cost accepted:** instance store is lost on *stop* as well as terminate, so an interrupted
  session rebuilds from the Dockerfile. That is ~45 unattended minutes, and it is the price of the
  reproducibility the Dockerfile provides anyway. It also forces the S3 publish to happen before
  release rather than "later".
- **General principle:** match storage durability to whether the data is reproducible. Paying to
  persist regenerable artifacts is a common and invisible waste.
- **Note:** an EBS root is not optional — EC2 will not boot from instance store. "No EBS" means
  minimal EBS, not none.

