"""
Step 1.2 gate check — run INSIDE the container before touching a model.

Every check here corresponds to a documented way this setup fails silently.
Run:  python src/scripts/verify_env.py
A clean pass is the definition of 1.2 being done.
"""
import os
import shutil
import sys

FAILURES = []


def check(name, fn):
    """Run one check with stdin closed, and never let a failure stop the others.

    stdin is redirected to /dev/null deliberately. `libero.libero` prompts interactively
    for dataset paths on first import; with a tty attached that reads as a hang, and
    inside this try/except it surfaced as an unrelated-looking error and cost hours on
    2026-08-26. With no stdin a prompt raises EOFError immediately and names itself.
    Any check in here that wants input is a check that would hang an unattended sweep.
    """
    try:
        with open(os.devnull) as devnull:
            saved, sys.stdin = sys.stdin, devnull
            try:
                detail = fn()
            finally:
                sys.stdin = saved
        print(f"  [PASS] {name}: {detail}")
    except Exception as e:  # noqa: BLE001 - we want every failure, not the first
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        FAILURES.append(name)


def torch_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not visible — did you pass --gpus all?"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    # OpenVLA-7B in bf16 is ~15 GB of weights.
    assert vram > 20, f"only {vram:.1f} GB VRAM — 7B bf16 will not fit"
    return f"{name}, {vram:.1f} GB, torch {torch.__version__}, sm_{cap[0]}{cap[1]}"


def transformers_pin():
    import transformers
    v = transformers.__version__
    # Hard requirement: OpenVLA's custom modeling code breaks above this.
    assert v == "4.40.1", f"got {v}, need exactly 4.40.1 — something upgraded it"
    return v


def flash_attn():
    import flash_attn
    return flash_attn.__version__


def host_ram():
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    # g6.xlarge has 16 GB. Loading a ~15 GB checkpoint through host RAM is the OOM risk.
    flag = "  <-- TIGHT for a 15 GB checkpoint load" if total < 24 else ""
    return f"{total:.1f} GB{flag}"


def disk():
    """Check the filesystem the checkpoint actually lands on, not '/'.

    Was `shutil.disk_usage("/")`, which measures the container root — i.e. the EBS volume.
    Per D-014 the box is disposable: HF_HOME is bind-mounted to the instance-store NVMe,
    which is where the ~15 GB checkpoint is written. On 2026-08-26 this produced a false
    FAIL at "37 GB free" while the actual target had 217 GB. A validation check pointed at
    the wrong resource is worse than no check — it sends you to debug a healthy system.
    """
    target = os.environ.get("HF_HOME") or "/"
    os.makedirs(target, exist_ok=True)
    free = shutil.disk_usage(target).free / 1e9
    # Checkpoint ~15 GB + LIBERO assets + headroom.
    assert free > 40, f"only {free:.0f} GB free at {target} — checkpoint download will fail"
    return f"{free:.0f} GB free at {target}"


def mujoco_egl():
    """The headless-rendering fight. If this passes, 1.2's hard part is over."""
    assert os.environ.get("MUJOCO_GL") == "egl", "MUJOCO_GL is not set to egl"
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=128, width=128)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    frame = renderer.render()
    assert frame.shape == (128, 128, 3), f"unexpected frame shape {frame.shape}"
    return f"mujoco {mujoco.__version__}, rendered {frame.shape} offscreen via EGL"


def libero_env():
    """Actually instantiate a LIBERO-Spatial task and render one observation."""
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    n = suite.n_tasks
    task = suite.get_task(0)
    return f"libero_spatial loaded, {n} tasks, task[0]='{task.language}'"


if __name__ == "__main__":
    print("Step 1.2 environment verification\n")
    check("torch + CUDA", torch_cuda)
    check("transformers pin", transformers_pin)
    check("flash-attn", flash_attn)
    check("host RAM", host_ram)
    check("disk space", disk)
    check("MuJoCo headless (EGL)", mujoco_egl)
    check("LIBERO benchmark", libero_env)

    print()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        print("1.2 is NOT done. Fix these before downloading a checkpoint.")
        sys.exit(1)
    print("All checks passed. 1.2 done -> proceed to 1.3 (one episode, one task).")
