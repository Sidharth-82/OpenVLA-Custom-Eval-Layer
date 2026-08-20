"""
Stage 0 — exercise the whole harness with a random policy. CPU only, no model, no cost.

Purpose: every part of the loop EXCEPT the policy gets validated here for free —
env construction, initial-state handling, observation shapes, the action convention,
episode termination, the success predicate, and per-step logging.

Anything this script gets wrong would otherwise be discovered at L4 prices, and would be
indistinguishable from "the policy is bad" once a real model is attached.

Run locally:
    MUJOCO_GL=osmesa python src/scripts/stage0_random_policy.py --task-id 0 --episodes 2

NOTE: written against the documented LIBERO API. Project rule is "verify live, do not
assume" — so this script INTROSPECTS and prints what it actually receives rather than
trusting the docs. If the API has drifted, the printout says so immediately.
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np

# Dummy action = hold still, gripper open. LIBERO objects need a few settling steps after
# set_init_state() before the first real action, or episode 1 behaves differently to the rest.
NOOP = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
SETTLE_STEPS = 10


def build_env(task_suite_name, task_id, resolution, seed):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution
    )
    env.seed(seed)
    init_states = suite.get_task_init_states(task_id)
    return env, task, init_states, suite.n_tasks


def describe(obs):
    """Print the actual observation structure. This is the arrow-labelling exercise, in code."""
    print("  observation keys -> shape / dtype / range")
    for k, v in sorted(obs.items()):
        arr = np.asarray(v)
        rng = f"[{arr.min():.3f}, {arr.max():.3f}]" if arr.size else "empty"
        print(f"    {k:<28} {str(arr.shape):<18} {str(arr.dtype):<10} {rng}")


def run_episode(env, init_state, max_steps, rng, log_path, episode_idx):
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(NOOP)

    if episode_idx == 0:
        describe(obs)

    success = False
    records = []
    t0 = time.time()
    for step in range(max_steps):
        # Random action in the documented [-1, 1] normalised action space.
        # 6 DoF delta pose + 1 gripper. If this range is wrong, the sim will show it.
        action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
        obs, reward, done, info = env.step(action)
        records.append(
            {
                "episode": episode_idx,
                "step": step,
                "action": action.round(4).tolist(),
                "reward": float(reward),
                "done": bool(done),
            }
        )
        if done:
            success = True  # LIBERO sets done on task success
            break

    elapsed = time.time() - t0
    with open(log_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return success, len(records), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=220)  # OpenVLA's cap for spatial
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stage0_random.jsonl")
    args = ap.parse_args()

    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    env, task, init_states, n_tasks = build_env(
        args.task_suite, args.task_id, args.resolution, args.seed
    )
    print(f"suite={args.task_suite}  tasks={n_tasks}  init_states={len(init_states)}")
    print(f"task[{args.task_id}].language = {task.language!r}")
    print(f"action_space = {getattr(env, 'action_dim', 'unknown')}\n")

    rng = np.random.default_rng(args.seed)
    successes, total_steps, total_time = 0, 0, 0.0
    for i in range(args.episodes):
        ok, steps, secs = run_episode(
            env, init_states[i % len(init_states)], args.max_steps, rng, args.out, i
        )
        successes += int(ok)
        total_steps += steps
        total_time += secs
        print(f"  episode {i}: success={ok}  steps={steps}  {secs:.1f}s")

    env.close()
    print(f"\n{successes}/{args.episodes} 'successes' with RANDOM actions.")
    print(f"sim throughput: {total_steps / total_time:.1f} steps/s (CPU, no policy)")
    print(f"per-step log -> {args.out}")
    print(
        "\nExpected result: ~0 successes. A random policy solving the task would mean the\n"
        "success predicate is wrong — which is exactly the kind of bug that makes a real\n"
        "policy look better than it is. That check is the point of this script."
    )


if __name__ == "__main__":
    main()
