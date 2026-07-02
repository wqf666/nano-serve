import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

"""
plot_kv_quant_results.py  —  Q3 Plotting Script
================================================
Read the JSON results produced by ``scripts/run_kv_quant_experiments.sh``
and generate comparison charts.

Usage
-----
    python benchmark/plot_kv_quant_results.py results/kv_quant_results.json

Output
------
Four PNG files in the same directory as the input JSON:
  - peak_kv_memory.png
  - memory_saving_ratio.png
  - tokens_per_second.png
  - output_match_rate.png

If matplotlib is not installed, the script falls back to printing ASCII
tables so the data is still inspectable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def pivot(results: list[dict], metric: str) -> tuple[list[str], dict[str, list[float | None]]]:
    """Pivot results into (workloads, {dtype: [values]})."""
    workloads: list[str] = sorted({r["workload"] for r in results if "workload" in r})
    dtypes: list[str] = sorted({r["dtype"] for r in results if "dtype" in r})

    data: dict[str, list[float | None]] = {}
    for dt in dtypes:
        vals: list[float | None] = []
        for wl in workloads:
            match = [r for r in results if r.get("workload") == wl and r.get("dtype") == dt]
            if match and match[0].get(metric) is not None:
                vals.append(match[0][metric])
            else:
                vals.append(None)
        data[dt] = vals

    return workloads, data


def compute_saving_ratio(results: list[dict]) -> tuple[list[str], dict[str, list[float | None]]]:
    """Compute memory saving ratio from original_bytes / quantized_bytes if
    the metric isn't already present."""
    workloads: list[str] = sorted({r["workload"] for r in results if "workload" in r})
    dtypes: list[str] = sorted({r["dtype"] for r in results if "dtype" in r})

    data: dict[str, list[float | None]] = {}
    for dt in dtypes:
        vals: list[float | None] = []
        for wl in workloads:
            match = [r for r in results if r.get("workload") == wl and r.get("dtype") == dt]
            if not match:
                vals.append(None)
                continue
            entry = match[0]
            # Prefer pre-computed ratio
            if entry.get("memory_saving_ratio") is not None:
                vals.append(entry["memory_saving_ratio"])
            elif entry.get("peak_kv_memory_mb") is not None and dt != "fp16":
                # Find fp16 baseline for same workload
                fp16_match = [r for r in results
                              if r.get("workload") == wl and r.get("dtype") == "fp16"
                              and r.get("peak_kv_memory_mb") is not None]
                if fp16_match:
                    baseline = fp16_match[0]["peak_kv_memory_mb"]
                    if baseline > 0:
                        vals.append(round(1.0 - entry["peak_kv_memory_mb"] / baseline, 4))
                    else:
                        vals.append(None)
                else:
                    vals.append(None)
            else:
                # fp16 baseline → 0% saving
                vals.append(0.0 if dt == "fp16" else None)
        data[dt] = vals

    return workloads, data


# ---------------------------------------------------------------------------
# Plotting (matplotlib)
# ---------------------------------------------------------------------------

BAR_WIDTH = 0.25
COLORS = {"fp16": "#4C72B0", "int8": "#DD8452", "fp8": "#55A868"}


def _bar_chart(
    workloads: list[str],
    data: dict[str, list[float | None]],
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(workloads))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (dt, vals) in enumerate(sorted(data.items())):
        numeric = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + i * BAR_WIDTH, numeric, BAR_WIDTH, label=dt, color=COLORS.get(dt, "#888"))
        # Annotate bars with values
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.2f}" if isinstance(v, float) else str(v),
                    ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x + BAR_WIDTH)
    ax.set_xticklabels(workloads)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _ascii_table(
    workloads: list[str],
    data: dict[str, list[float | None]],
    title: str,
    ylabel: str,
) -> None:
    """Fallback when matplotlib is unavailable."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    header = f"{'Workload':<22}" + "".join(f"{dt:>12}" for dt in sorted(data))
    print(header)
    print("-" * len(header))
    for i, wl in enumerate(workloads):
        row = f"{wl:<22}"
        for dt in sorted(data):
            v = data[dt][i]
            row += f"{v:>12.4f}" if v is not None else f"{'N/A':>12}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_CONFIGS = [
    {
        "metric": "peak_kv_memory_mb",
        "title": "Peak KV Cache Memory (MB) — Lower is Better",
        "ylabel": "Peak KV Memory (MB)",
        "filename": "peak_kv_memory.png",
    },
    {
        "metric": "_saving_ratio",  # computed
        "title": "Memory Saving Ratio vs FP16 — Higher is Better",
        "ylabel": "Saving Ratio (1 - quant/baseline)",
        "filename": "memory_saving_ratio.png",
    },
    {
        "metric": "throughput_tokens_per_sec",
        "title": "Throughput (tokens/sec) — Higher is Better",
        "ylabel": "Tokens / Second",
        "filename": "tokens_per_second.png",
    },
    {
        "metric": "output_match_rate",
        "title": "Output Match Rate vs FP16 Baseline — Higher is Better",
        "ylabel": "Match Rate (0–1)",
        "filename": "output_match_rate.png",
    },
]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python benchmark/plot_kv_quant_results.py <results.json>")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: file not found: {results_path}")
        sys.exit(1)

    results = load_results(str(results_path))
    if not results:
        print("Warning: results file is empty. Nothing to plot.")
        sys.exit(0)

    output_dir = results_path.parent

    # Check matplotlib availability
    try:
        import matplotlib  # noqa: F401
        has_mpl = True
    except ImportError:
        has_mpl = False
        print("WARNING: matplotlib not installed. Falling back to ASCII tables.\n")

    for cfg in METRIC_CONFIGS:
        metric = cfg["metric"]

        if metric == "_saving_ratio":
            workloads, data = compute_saving_ratio(results)
        else:
            workloads, data = pivot(results, metric)

        if not workloads:
            print(f"  Skipping {cfg['filename']}: no data.")
            continue

        if has_mpl:
            out_path = str(output_dir / cfg["filename"])
            _bar_chart(workloads, data, cfg["title"], cfg["ylabel"], out_path)
        else:
            _ascii_table(workloads, data, cfg["title"], cfg["ylabel"])

    print("\nDone.")


if __name__ == "__main__":
    main()
