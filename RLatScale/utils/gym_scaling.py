"""
gym_scaling.py
--------------
Throughput scaling benchmark: steps/s vs number of parallel environments
for Gym environments (CPU Gymnasium vs GPU Gymnax).

Sweeps num_envs on a log scale up to 10 M.  CPU is capped at a lower
practical limit; GPU backends continue until OOM or the 10 M ceiling.

Outputs saved to results/gym_scaling_{timestamp}/
    scaling.png    — log-log throughput vs num_envs (all backend/impl combos)
    scaling.csv    — raw (backend, impl, num_envs, steps_per_second) data

Usage
-----
    python -m RLatScale.utils.gym_scaling
"""

from __future__ import annotations

import csv
import dataclasses
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from RLatScale.algo.config import Config, GPUConfig
from RLatScale.gym_test import cpu_test, gpu_test
from RLatScale.gym_test.cpu_test import ResourceMonitor


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

_PROBE_ROLLOUTS = 5       # rollouts per probe point (amortises JIT warmup)
_PROBE_ENV      = "CartPole-v1"   # fastest env; isolates raw throughput

# Log-scale sweeps (powers of 2, capped per backend)
_CPU_SWEEP: list[int] = [2**i for i in range(10)]          # 1 → 512
_GPU_SWEEP: list[int] = [2**i for i in range(20)] + [1_000_000]  # 1 → 1 M


# ---------------------------------------------------------------------------
# Colour / style palette
# ---------------------------------------------------------------------------

_STYLE: dict[tuple[str, str], dict] = {
    ("cpu", "linen"): {"color": "#1D4ED8", "linestyle": "-",  "marker": "o", "label": "CPU · Linen"},
    ("cpu", "nnx"):   {"color": "#60A5FA", "linestyle": "--", "marker": "s", "label": "CPU · NNX"},
    ("cpu", "ion"):   {"color": "#1D4ED8", "linestyle": "-",  "marker": "o", "label": "CPU"},
    ("gpu", "linen"): {"color": "#EA580C", "linestyle": "-",  "marker": "o", "label": "GPU · Linen"},
    ("gpu", "nnx"):   {"color": "#FB923C", "linestyle": "--", "marker": "s", "label": "GPU · NNX"},
    ("gpu", "ion"):   {"color": "#EA580C", "linestyle": "-",  "marker": "o", "label": "GPU"},
}


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------

def _probe(
    base_config: Config,
    run_fn,
    env_id: str,
    impl: str,
    n_envs: int,
) -> float | None:
    """Run a short throughput probe. Returns steps/s, or None on failure."""
    probe = dataclasses.replace(
        base_config,
        num_envs=n_envs,
        total_timesteps=_PROBE_ROLLOUTS * n_envs * base_config.num_steps,
        hardware_tag="",
    )
    monitor = ResourceMonitor()
    try:
        metrics, _ = run_fn(probe, env_id, impl, 1, monitor)
        return float(metrics["mean_steps_per_second"])
    except Exception as exc:
        print(f"    [SKIP] {exc!r}")
        return None


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def sweep(
    backend: str,
    base_config: Config,
    run_fn,
    env_id: str,
    impl: str,
    counts: list[int],
) -> dict[int, float]:
    """Probe throughput for each num_envs in counts; stop at first failure."""
    results: dict[int, float] = {}
    for n in counts:
        print(f"  [{backend.upper()}/{impl}] {n:>10,} envs …", end=" ", flush=True)
        sps = _probe(base_config, run_fn, env_id, impl, n)
        if sps is None:
            print()
            break
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
    ax.set_title(f"Throughput Scaling — {_PROBE_ENV}")
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
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"gym_scaling_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {out_dir}\n{'═' * 60}")

    cpu_base = Config()
    gpu_base = GPUConfig()

    all_results: dict[tuple[str, str], dict[int, float]] = {}

    for impl in ("ion",):
        print(f"\n── CPU / {impl} ──")
        all_results[("cpu", impl)] = sweep(
            "cpu", cpu_base, cpu_test.run_experiment, _PROBE_ENV, impl, _CPU_SWEEP
        )

    for impl in ("ion",):
        print(f"\n── GPU / {impl} ──")
        all_results[("gpu", impl)] = sweep(
            "gpu", gpu_base, gpu_test.run_experiment, _PROBE_ENV, impl, _GPU_SWEEP
        )

    print(f"\n{'═' * 60}\nGenerating outputs …")
    plot_scaling(all_results, out_dir)
    save_csv(all_results, out_dir)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
