# OpenVLA Custom Eval Layer

**An evaluation layer for vision-language-action policies. Not a model, not a training run — a harness that answers: *when does this policy break, and is the difference real?***

---

## The question

Standard VLA benchmarks report a single number: success rate on a fixed task suite. That number tells you a policy works. It does not tell you **where it stops working**, or whether a difference between two policies is a real effect or sampling noise.

This project builds the layer that answers those two questions:

1. **Under what perturbation does the policy degrade?** v1 perturbs the *language* half of a vision-language-action model — the same task, the same scene, a paraphrased instruction. That separates genuine language grounding from scene pattern-matching, and it is barely probed by the standard benchmark.
2. **Is the delta real?** Every reported success rate carries a Wilson confidence interval. No number ships without one.

**No training happens in this project.** It is inference-only by design — see [D-002](./DECISIONS.md).

---

## Status

> **Design complete. Implementation in progress — Step 1 of 6.**
>
> The reproduction gate ([D-013](./DECISIONS.md)) is being run now: OpenVLA on unmodified LIBERO-Spatial must match the published success rate within a tolerance fixed *before* the run. Nothing downstream begins until it passes.

This is deliberate ordering, not a stalled project. Step 1 is a hard gate. For an eval harness, **"the policy is bad" and "my harness is broken" are indistinguishable without a truth anchor** — so the anchor gets established first, against a published baseline, before a single perturbation is introduced.

| Step | Done means | State |
|---|---|---|
| **1. Reproduce** | OpenVLA on unmodified LIBERO matches published SR within the D-013 gate | **In progress** |
| 2. Runner | YAML scenario spec, seeded, containerized, JSONL → S3. Same seed twice = same result | Designed |
| 3. Paraphrase axis | 4 instruction levels, version-controlled; sweep executes end to end | Designed |
| 4. Failure taxonomy | Every failed episode auto-classified: grasp miss / wrong object / collision / timeout / drift | Designed |
| 5. Slice + stats | Success rate per slice with Wilson CIs. Answers "is this delta real?" | Designed |
| 6. Regression + report | Two policy versions, one command, one HTML report | Designed |

---

## The reasoning is the product

**→ [`DECISIONS.md`](./DECISIONS.md) — thirteen design decisions, written as they were made.**

For an evaluation project this is not documentation *about* the work; it is the work. **A number nobody can interrogate is not a result.** Each entry records what was chosen, what it was chosen over, why, and — where it matters — the cost accepted and the threat to validity.

A few that carry the design:

| | Decision | The reasoning in one line |
|---|---|---|
| **D-003** | LIBERO over ManiSkill3 | Published baselines from multiple VLA papers give a truth anchor; without one, harness bugs and policy failures are indistinguishable. Traded ManiSkill3's GPU-parallel throughput for verifiability |
| **D-004** | Depth over breadth | 50 trials/cell on one axis, not 15 across eight. At n=15 a genuine 10-point drop is inside the noise — a wide sweep produces a colourful chart of undefendable conclusions |
| **D-007** | Five highest-baseline tasks | Measurement headroom: a task at 20% baseline cannot demonstrate degradation. **Stated as a threat to validity** — this is a biased sample of the suite, and the report says so |
| **D-013** | Reproduction gate at ±5pp | Derived from the standard error at n=500 against the paper's n=1500, combined in quadrature, ×1.96. **Fixed before the run**, so the gate cannot be moved to fit the result |

---

## Setup

| Piece | Choice |
|---|---|
| Task suite | LIBERO (MuJoCo / robosuite), LIBERO-Spatial |
| Policy A | OpenVLA-7B, official LIBERO-finetuned checkpoint, bf16 |
| Policy B | OpenVLA-7B, 4-bit quantized — same weights, one flag |
| Perturbation axis | Language instruction paraphrase (v1 only) |
| Trial semantics | 50 distinct LIBERO initial states, deterministic decoding, reused across every cell |
| Recorded per episode | Binary success **+ dense progress** — min distance to target, grasp flag, lift flag |
| Trials | 50 per cell |
| Execution | EC2 GPU, Docker, results → S3 |

**Sweep budget, fixed in advance:**

```
Policy A:  5 tasks x 4 paraphrase levels x 50 trials = 1,000 episodes  (~25 GPU-hr)
Policy B:  5 tasks x 2 levels           x 50 trials =   500 episodes  (~12 GPU-hr)
Total ~1,500 episodes / ~37 GPU-hr
```

---

## Architecture note

The runner takes a **perturbation object that transforms the scenario spec**. It knows nothing about paraphrasing, jitter, or distractors ([D-009](./DECISIONS.md)).

Adding an axis is a new module plus config — never a rewrite of the run loop. That is the line between an eval *layer* and an eval *script*, and axes ship strictly one at a time ([D-008](./DECISIONS.md)): v1.5 adds object pose jitter, v2 adds distractor objects. Neither starts until v1 ships.

---

## Layout

```
src/docker/       Dockerfile — pinned to the exact environment the published numbers used
src/scripts/      verify_env.py (step 1.2 gate check), stage0_random_policy.py
01 Steps/         per-step working notes, live
00 Design/        design notes and decision working
DECISIONS.md      the reasoning
results/          JSONL episode records
```

**Environment is pinned, not approximate.** `transformers==4.40.1` is a hard pin — OpenVLA ships custom modeling code that breaks on later versions. `--center_crop True` is required at eval, because training used random-crop augmentation; omitting it is the single most common silent reproduction failure.

---

## Honest limitations

- **Sim only.** LIBERO is MuJoCo. No real-robot validation, and no claim of one.
- **One axis in v1.** Language paraphrase only. Camera-extrinsics jitter maps more directly to real deployment and is parked as the strongest v1.5 candidate ([D-005](./DECISIONS.md)).
- **The task selection is biased on purpose.** Choosing the five highest-baseline tasks buys measurement headroom and costs generality. The claim is scoped accordingly: *paraphrase degrades the tasks the policy is best at.*
- **No training track record.** Inference-only is a deliberate trade ([D-002](./DECISIONS.md)), and it means this project demonstrates nothing about training pipelines.
