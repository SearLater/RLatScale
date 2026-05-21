"""
gym_eval.py
-----------
Run CPU and GPU PPO benchmarks back-to-back and write comparison outputs.

CPU backend : gym_test.cpu_test  (Gymnasium, Python-loop, Config)
GPU backend : gym_test.gpu_test  (Gymnax, JAX scan,       GPUConfig)

Both Linen and NNX implementations are run for each backend so that four
learning curves appear on each environment plot.

Outputs saved to results/gym_eval_{timestamp}/
    {env}_curves.png     — IQM learning curves, CPU vs GPU per environment
    throughput.png       — Steps/s bar chart across all (backend, impl, env) combos
    summary.csv          — Machine-readable metrics table
    summary_table.png    — Rendered visual summary table
    runs/                — Individual run artefacts (config, raw seeds, metrics, plots)

Usage
-----
    python -m RLatScale.utils.gym_eval
"""

from __future__ import annotations

import csv
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from RLatScale.algo.config import Config, GPUConfig
from RLatScale.gym_test import cpu_test, gpu_test
from RLatScale.gym_test.cpu_test import ResourceMonitor, _print_summary, save_run


# ---------------------------------------------------------------------------
# Colour / style palette
# ---------------------------------------------------------------------------

_STYLE: dict[tuple[str, str], dict] = {
    ("cpu", "linen"): {"color": "#1D4ED8", "linestyle": "-",  "label": "CPU · Linen"},
    ("cpu", "nnx"):   {"color": "#60A5FA", "linestyle": "--", "label": "CPU · NNX"},
    ("cpu", "ion"):   {"color": "#1E8A3E", "linestyle": ":",  "label": "CPU · Ion"},
    ("gpu", "linen"): {"color": "#EA580C", "linestyle": "-",  "label": "GPU · Linen"},
    ("gpu", "nnx"):   {"color": "#FB923C", "linestyle": "--", "label": "GPU · NNX"},
    ("gpu", "ion"):   {"color": "#16A34A", "linestyle": ":",  "label": "GPU · Ion"},
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class RunResult(NamedTuple):
    backend:      str        # "cpu" or "gpu"
    metrics:      dict
    seed_results: list[dict]
    config:       Config


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_backend(
    backend: str,
    config: Config,
    run_experiment_fn,
    out_dir: Path,
) -> list[RunResult]:
    """Run every (env, impl) combo for one backend; persist individual runs."""
    results: list[RunResult] = []
    for env_id in config.envs:
        for impl in config.impls:
            print(f"\n[{backend.upper()}] {impl}/{env_id} × {config.num_seeds} seeds …")
            monitor = ResourceMonitor()
            metrics, seed_results = run_experiment_fn(
                config, env_id, impl, config.num_seeds, monitor
            )
            metrics["backend"] = backend
            _print_summary(metrics)
            save_run(metrics, seed_results, config, base_dir=out_dir / "runs")
            results.append(RunResult(backend, metrics, seed_results, config))
    return results


def run_all(cpu_config: Config, gpu_config: Config, out_dir: Path) -> list[RunResult]:
    cpu_results = _run_backend("cpu", cpu_config, cpu_test.run_experiment, out_dir)
    gpu_results = _run_backend("gpu", gpu_config, gpu_test.run_experiment, out_dir)
    return cpu_results + gpu_results


# ---------------------------------------------------------------------------
# Comparison learning-curve plots
# ---------------------------------------------------------------------------

def plot_curves(results: list[RunResult], out_dir: Path) -> None:
    """One PNG per environment — IQM ± P25/P75 for every (backend, impl) combo."""
    envs = sorted({r.metrics["env"] for r in results})

    for env_id in envs:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        for result in results:
            m = result.metrics
            if m["env"] != env_id:
                continue

            style = _STYLE.get((result.backend, m["impl"]), {})
            steps = np.array(m["steps_axis"]) / 1e6
            iqm   = np.array(m["iqm_curve"])
            p25   = np.array(m["p25_curve"])
            p75   = np.array(m["p75_curve"])

            ax.fill_between(steps, p25, p75, color=style["color"], alpha=0.12)
            ax.plot(
                steps, iqm,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=style.get("label", f"{result.backend}/{m['impl']}"),
            )

        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="Threshold")
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylabel("Normalised return (IQM)")
        ax.set_title(env_id)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0)
        fig.tight_layout()

        path = out_dir / f"{env_id.replace('-', '_')}_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Throughput bar chart
# ---------------------------------------------------------------------------

def plot_throughput(results: list[RunResult], out_dir: Path) -> None:
    """Grouped bar chart: steps/s for each (backend, impl) combo per environment."""
    envs   = sorted({r.metrics["env"] for r in results})
    combos = [k for k in _STYLE if any(
        r.backend == k[0] and r.metrics["impl"] == k[1] for r in results
    )]

    n_envs, n_combos = len(envs), len(combos)
    x         = np.arange(n_envs)
    bar_width = 0.8 / n_combos

    fig, ax = plt.subplots(figsize=(max(6, n_envs * 2.5), 4.5))

    for i, (backend, impl) in enumerate(combos):
        style = _STYLE[(backend, impl)]
        sps_vals = []
        for env_id in envs:
            match = next(
                (r for r in results
                 if r.backend == backend
                 and r.metrics["impl"] == impl
                 and r.metrics["env"] == env_id),
                None,
            )
            sps_vals.append(match.metrics["mean_steps_per_second"] if match else 0.0)

        offset = (i - n_combos / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, sps_vals, bar_width,
            label=style["label"],
            color=style["color"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, sps_vals):
            if val > 0:
                label = f"{val/1000:.1f}k" if val >= 1000 else f"{val:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    label,
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=9)
    ax.set_ylabel("Steps per second")
    ax.set_title("Throughput: CPU vs GPU")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt(val: float | str, fmt: str = ".3f") -> str:
    if isinstance(val, float) and np.isnan(val):
        return "—"
    if isinstance(val, float):
        return format(val, fmt)
    return str(val)


def save_table(results: list[RunResult], out_dir: Path) -> None:
    """Write summary.csv and summary_table.png."""
    rows = []
    for r in results:
        m = r.metrics
        rows.append({
            "Backend":      r.backend.upper(),
            "Impl":         m["impl"],
            "Env":          m["env"],
            "AUC mean":     _fmt(m["auc_mean"]),
            "IQM":          _fmt(m["iqm"]),
            "IQM 95% CI":   f"[{_fmt(m['iqm_ci_lo'])}, {_fmt(m['iqm_ci_hi'])}]",
            "Steps/s":      f"{m['mean_steps_per_second']:,.0f}",
            "Time→thr (s)": _fmt(m["mean_time_to_threshold"], ".1f"),
            "% thr":        f"{m['pct_reached_threshold'] * 100:.0f}%",
        })

    if not rows:
        return

    # CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {csv_path}")

    # Visual table PNG
    headers   = list(rows[0].keys())
    cell_data = [[row[h] for h in headers] for row in rows]
    n_rows    = len(cell_data)

    fig, ax = plt.subplots(figsize=(15, max(2.0, n_rows * 0.45 + 1.2)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(headers))))

    for col in range(len(headers)):
        tbl[(0, col)].set_facecolor("#1E40AF")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    for row in range(1, n_rows + 1):
        bg = "#EFF6FF" if row % 2 == 0 else "white"
        for col in range(len(headers)):
            tbl[(row, col)].set_facecolor(bg)

    ax.set_title("Gym Benchmark Summary", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()

    table_path = out_dir / "summary_table.png"
    fig.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {table_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"gym_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {out_dir}\n{'═' * 60}")

    cpu_config = dataclasses.replace(Config(),    impls=("ion",), hardware_tag="")
    gpu_config = dataclasses.replace(GPUConfig(), impls=("ion",), hardware_tag="")

    results = run_all(cpu_config, gpu_config, out_dir)

    print(f"\n{'═' * 60}\nGenerating outputs …")
    plot_curves(results, out_dir)
    plot_throughput(results, out_dir)
    save_table(results, out_dir)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
