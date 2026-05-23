"""
mujoco_scaling.py
-----------------
Throughput scaling benchmark: steps/s vs number of parallel environments
for MuJoCo environments (CPU Gymnasium vs Brax vs MJX).

Sweeps num_envs on a log scale up to 10 M.  CPU is capped at a low limit
(SyncVectorEnv is sequential so MuJoCo sim time grows linearly with envs).
Brax and MJX backends continue until OOM or the 10 M ceiling.

A fixed num_steps (_PROBE_STEPS) overrides each backend's default so that
steps/s is measured under the same rollout geometry across all backends.

Outputs saved to results/mujoco_scaling_{timestamp}/
    scaling.png    — log-log throughput vs num_envs (all backend/impl combos)
    scaling.csv    — raw (backend, impl, num_envs, steps_per_second) data

Usage
-----
    python -m RLatScale.utils.mujoco_scaling
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RLatScale.algo.config import BraxConfig, MjxConfig, MuJoCoConfig
from RLatScale.mujoco_test import brax_test, cpu_test, mjx_test
from RLatScale.gym_test.cpu_test import ResourceMonitor


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

_PROBE_ROLLOUTS = 3     # rollouts per probe point
_PROBE_STEPS    = 64    # override num_steps for all backends (keeps probes fast)

# CPU MuJoCo sim is slow per step; keep sweep small to avoid hour-long runs.
_CPU_SWEEP:      list[int] = [2**i for i in range(7)]       # 1 → 64
# Brax crashes with SIGABRT for very small env counts due to its physics backend.
# Start from 8 — uninteresting for a GPU scaling study anyway.
_BRAX_GPU_SWEEP: list[int] = [2**i for i in range(3, 21)]  # 8 → 1,048,576 (2^20)
_MJX_GPU_SWEEP:  list[int] = [2**i for i in range(17)]     # 1 → 65,536 (2^16)

# Environment identifiers per backend
_CPU_ENV  = "HalfCheetah-v4"   # Gymnasium name
_BRAX_ENV = "halfcheetah"      # Brax / MJX short name
_MJX_ENV  = "halfcheetah"


# ---------------------------------------------------------------------------
# Colour / style palette
# ---------------------------------------------------------------------------

_STYLE: dict[tuple[str, str], dict] = {
    ("cpu",  "linen"): {"color": "#1D4ED8", "linestyle": "-",  "marker": "o", "label": "CPU · Linen"},
    ("cpu",  "nnx"):   {"color": "#60A5FA", "linestyle": "--", "marker": "s", "label": "CPU · NNX"},
    ("cpu",  "ion"):   {"color": "#1D4ED8", "linestyle": "-",  "marker": "o", "label": "CPU"},
    ("brax", "linen"): {"color": "#15803D", "linestyle": "-",  "marker": "o", "label": "Brax · Linen"},
    ("brax", "nnx"):   {"color": "#4ADE80", "linestyle": "--", "marker": "s", "label": "Brax · NNX"},
    ("brax", "ion"):   {"color": "#15803D", "linestyle": "-",  "marker": "o", "label": "Brax"},
    ("mjx",  "linen"): {"color": "#EA580C", "linestyle": "-",  "marker": "o", "label": "MJX · Linen"},
    ("mjx",  "nnx"):   {"color": "#FB923C", "linestyle": "--", "marker": "s", "label": "MJX · NNX"},
    ("mjx",  "ion"):   {"color": "#EA580C", "linestyle": "-",  "marker": "o", "label": "MJX"},
}


# ---------------------------------------------------------------------------
# Single-probe subprocess entry point
# ---------------------------------------------------------------------------

def _run_single_probe(backend: str, env_id: str, impl: str, n_envs: int) -> None:
    """Run one probe point and print the result as JSON. Called via --probe CLI flag."""
    _CONFIG = {"cpu": MuJoCoConfig, "brax": BraxConfig, "mjx": MjxConfig}
    _RUN_FN = {
        "cpu":  cpu_test.run_experiment,
        "brax": brax_test.run_experiment,
        "mjx":  mjx_test.run_experiment,
    }
    probe = dataclasses.replace(
        _CONFIG[backend](),
        num_envs=n_envs,
        num_steps=_PROBE_STEPS,
        total_timesteps=_PROBE_ROLLOUTS * n_envs * _PROBE_STEPS,
        hardware_tag="",
    )
    monitor = ResourceMonitor()
    metrics, _ = _RUN_FN[backend](probe, env_id, impl, 1, monitor)
    print(json.dumps({"sps": float(metrics["mean_steps_per_second"])}))


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------

def _probe(base_config, run_fn, env_id: str, impl: str, n_envs: int) -> float | None:
    """Run a short throughput probe in a fresh interpreter. Returns steps/s, or None.

    Uses subprocess.run so no fork() occurs in the parent — loading Brax/MJX
    environments multiple times in the same process triggers a glibc double-free.
    Each probe is a completely fresh Python process with no inherited C state.
    """
    _BACKEND = {MuJoCoConfig: "cpu", BraxConfig: "brax", MjxConfig: "mjx"}
    backend  = _BACKEND[type(base_config)]

    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    result = subprocess.run(
        [sys.executable, "-m", "RLatScale.utils.mujoco_scaling",
         "--probe", backend, env_id, impl, str(n_envs)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()
        last = err[-1] if err else "(no stderr)"
        print(f"    [SKIP] (exit {result.returncode}) {last}")
        return None
    try:
        return json.loads(result.stdout.strip())["sps"]
    except (json.JSONDecodeError, KeyError):
        print(f"    [SKIP] unexpected output: {result.stdout!r}")
        return None


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def sweep(
    backend: str,
    base_config,
    run_fn,
    env_id: str,
    impl: str,
    counts: list[int],
) -> dict[int, float]:
    """Probe throughput for each num_envs in counts; skip failures, stop on OOM."""
    results: dict[int, float] = {}
    consecutive_failures = 0
    for n in counts:
        print(f"  [{backend.upper()}/{impl}] {n:>10,} envs …", end=" ", flush=True)
        sps = _probe(base_config, run_fn, env_id, impl, n)
        if sps is None:
            print()
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break  # three in a row → likely OOM ceiling, stop
            continue
        consecutive_failures = 0
        print(f"{sps:>12,.0f} steps/s")
        results[n] = sps
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_scaling(
    all_results: dict[tuple[str, str], dict[int, float]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for (backend, impl), data in all_results.items():
        if not data:
            continue
        style = _STYLE.get((backend, impl), {})
        xs = list(data.keys())
        ys = list(data.values())
        ax.loglog(
            xs, ys,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker=style["marker"],
            markersize=5,
            label=style.get("label", f"{backend}/{impl}"),
        )

    ax.set_xlabel("Number of parallel environments")
    ax.set_ylabel("Throughput (steps/s)")
    ax.set_title("Throughput Scaling — HalfCheetah (CPU vs Brax vs MJX)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    fig.tight_layout()

    path = out_dir / "scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(
    all_results: dict[tuple[str, str], dict[int, float]],
    out_dir: Path,
) -> None:
    path = out_dir / "scaling.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "impl", "num_envs", "steps_per_second"])
        for (backend, impl), data in all_results.items():
            for n_envs, sps in data.items():
                writer.writerow([backend, impl, n_envs, f"{sps:.2f}"])
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe", nargs=4, metavar=("BACKEND", "ENV", "IMPL", "N_ENVS"))
    args, _ = parser.parse_known_args()
    if args.probe:
        backend, env_id, impl, n_envs = args.probe
        _run_single_probe(backend, env_id, impl, int(n_envs))
        return

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"mujoco_scaling_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {out_dir}\n{'═' * 60}")

    cpu_base  = MuJoCoConfig()
    brax_base = BraxConfig()
    mjx_base  = MjxConfig()

    all_results: dict[tuple[str, str], dict[int, float]] = {}

    for impl in ("ion",):
        print(f"\n── CPU / {impl} ──")
        all_results[("cpu", impl)] = sweep(
            "cpu", cpu_base, cpu_test.run_experiment, _CPU_ENV, impl, _CPU_SWEEP
        )

    for impl in ("ion",):
        print(f"\n── Brax / {impl} ──")
        all_results[("brax", impl)] = sweep(
            "brax", brax_base, brax_test.run_experiment, _BRAX_ENV, impl, _BRAX_GPU_SWEEP
        )

    for impl in ("ion",):
        print(f"\n── MJX / {impl} ──")
        all_results[("mjx", impl)] = sweep(
            "mjx", mjx_base, mjx_test.run_experiment, _MJX_ENV, impl, _MJX_GPU_SWEEP
        )

    print(f"\n{'═' * 60}\nGenerating outputs …")
    plot_scaling(all_results, out_dir)
    save_csv(all_results, out_dir)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
