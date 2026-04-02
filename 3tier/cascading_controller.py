#!/usr/bin/env python3
"""
Cascading controller for 3-tier miss rate predictions.

CascadingController: given per-window predictions of miss_rate for nano/small/medium,
picks the cheapest model whose predicted miss rate is below a threshold.

Usage:
    python cascading_controller.py                     # uses 3tier/predictions_3tier.csv
    python cascading_controller.py --predictions PATH  # custom predictions file
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import hardware constants
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
from controller_autosearch.models import MODELS, energy_per_window

E_NANO = energy_per_window(0)    # 1280.4 mJ
E_SMALL = energy_per_window(1)   # 1929.75 mJ
E_MEDIUM = energy_per_window(2)  # 3726.9 mJ
ENERGIES = np.array([E_NANO, E_SMALL, E_MEDIUM])
MODEL_NAMES = ["nano", "small", "medium"]

TRAIN_FRAC = 0.60


def cascading_select(pred_nano, pred_small, pred_medium, threshold):
    """For each window, pick the cheapest model with predicted miss rate < threshold.

    Returns: array of model indices (0=nano, 1=small, 2=medium).
    """
    n = len(pred_nano)
    selections = np.full(n, 2, dtype=int)  # default: medium
    selections[pred_small < threshold] = 1
    selections[pred_nano < threshold] = 0
    return selections


def oracle_select(true_nano, true_small, true_medium, threshold):
    """Oracle: cheapest model whose TRUE miss rate < threshold."""
    n = len(true_nano)
    selections = np.full(n, 2, dtype=int)
    selections[true_small < threshold] = 1
    selections[true_nano < threshold] = 0
    return selections


def evaluate_selections(selections, true_nano, true_small, true_medium):
    """Given model selections and true miss rates, compute metrics."""
    n = len(selections)
    true_rates = np.array([true_nano, true_small, true_medium])  # (3, n)

    # Per-window: true miss rate of the selected model
    selected_miss_rates = true_rates[selections, np.arange(n)]
    detection_rate = 1.0 - selected_miss_rates.mean()

    # Energy
    selected_energies = ENERGIES[selections]
    avg_energy = selected_energies.mean()
    savings_vs_medium = (1.0 - avg_energy / E_MEDIUM) * 100.0

    # Model distribution
    dist = {}
    for i, name in enumerate(MODEL_NAMES):
        dist[name] = float(np.mean(selections == i) * 100.0)

    return {
        "detection_rate": detection_rate,
        "avg_energy_mj": avg_energy,
        "savings_pct": savings_vs_medium,
        "mean_miss_rate": float(selected_miss_rates.mean()),
        "model_distribution": dist,
    }


def search_cascading(pred_nano, pred_small, pred_medium,
                     true_nano, true_small, true_medium,
                     thresholds=None):
    """Sweep max_acceptable_miss_rate and evaluate cascading controller."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.50, 20)

    results = []
    for thresh in thresholds:
        sels = cascading_select(pred_nano, pred_small, pred_medium, thresh)
        m = evaluate_selections(sels, true_nano, true_small, true_medium)
        m["threshold"] = float(thresh)
        results.append(m)

    return results


def plot_accuracy_vs_energy(results, ref_points, save_path):
    """Plot detection rate vs energy savings with reference points."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Cascading controller curve
    x = [r["savings_pct"] for r in results]
    y = [r["detection_rate"] * 100 for r in results]
    ax.plot(x, y, "o-", color="#2196F3", linewidth=2, markersize=6,
            label="Cascading Controller", zorder=5)

    # Annotate thresholds
    for i, r in enumerate(results):
        if i % 3 == 0 or i == len(results) - 1:
            ax.annotate(f'{r["threshold"]:.2f}',
                        (r["savings_pct"], r["detection_rate"] * 100),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color="#666")

    # Reference points
    markers = {"AlwaysNano": ("^", "#F44336"), "BestFixed(small)": ("s", "#FF9800"),
               "AlwaysMedium": ("D", "#4CAF50"), "Oracle": ("*", "#9C27B0")}
    for name, vals in ref_points.items():
        if name not in markers:
            continue
        mk, col = markers[name]
        ax.scatter(vals["savings_pct"], vals["detection_rate"] * 100,
                   marker=mk, s=150, color=col, edgecolors="black",
                   linewidths=1, zorder=10, label=name)

    # Oracle curve (if available)
    if "oracle_curve" in ref_points:
        ox = [r["savings_pct"] for r in ref_points["oracle_curve"]]
        oy = [r["detection_rate"] * 100 for r in ref_points["oracle_curve"]]
        ax.plot(ox, oy, "--", color="#9C27B0", linewidth=1.5, alpha=0.6,
                label="Oracle Frontier", zorder=4)

    ax.set_xlabel("Energy Savings vs AlwaysMedium (%)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("3-Tier Cascading Controller: Detection Rate vs Energy Savings",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 70)
    ax.set_ylim(40, 102)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str,
                        default=str(SCRIPT_DIR / "predictions_3tier.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    n = len(df)
    split_idx = int(n * TRAIN_FRAC)

    # Use test split only
    test = df.iloc[split_idx:]
    pred_nano = test["pred_nano"].values
    pred_small = test["pred_small"].values
    pred_medium = test["pred_medium"].values
    true_nano = test["true_nano"].values
    true_small = test["true_small"].values
    true_medium = test["true_medium"].values
    n_test = len(test)
    print(f"  Test split: {n_test} windows (of {n} total)", file=sys.stderr)

    # ── Sweep cascading controller ──
    thresholds = np.linspace(0.05, 0.50, 20)
    results = search_cascading(pred_nano, pred_small, pred_medium,
                               true_nano, true_small, true_medium,
                               thresholds)

    # ── Reference points ──
    # AlwaysNano
    sels_nano = np.zeros(n_test, dtype=int)
    ref_nano = evaluate_selections(sels_nano, true_nano, true_small, true_medium)
    ref_nano["threshold"] = None

    # BestFixed(small)
    sels_small = np.ones(n_test, dtype=int)
    ref_small = evaluate_selections(sels_small, true_nano, true_small, true_medium)
    ref_small["threshold"] = None

    # AlwaysMedium
    sels_medium = np.full(n_test, 2, dtype=int)
    ref_medium = evaluate_selections(sels_medium, true_nano, true_small, true_medium)
    ref_medium["threshold"] = None

    # Oracle at the "best" threshold (use median threshold from sweep)
    # Actually compute oracle for each threshold to get oracle curve
    oracle_results = []
    for thresh in thresholds:
        sels_oracle = oracle_select(true_nano, true_small, true_medium, thresh)
        m_oracle = evaluate_selections(sels_oracle, true_nano, true_small, true_medium)
        m_oracle["threshold"] = float(thresh)
        oracle_results.append(m_oracle)

    # Use oracle at threshold=0.30 as the reference point
    oracle_ref_idx = min(range(len(thresholds)),
                         key=lambda i: abs(thresholds[i] - 0.30))
    ref_oracle = oracle_results[oracle_ref_idx]

    ref_points = {
        "AlwaysNano": ref_nano,
        "BestFixed(small)": ref_small,
        "AlwaysMedium": ref_medium,
        "Oracle": ref_oracle,
        "oracle_curve": oracle_results,
    }

    # ── Print table ──
    print(f"\n{'='*105}")
    print(f"  Cascading Controller Results (test split, n={n_test})")
    print(f"{'='*105}")
    print(f"{'Threshold':>10s} | {'DetRate%':>8s} | {'Savings%':>8s} | "
          f"{'AvgEnergy':>10s} | {'MissRate':>8s} | "
          f"{'Nano%':>6s} {'Small%':>6s} {'Med%':>6s}")
    print("-" * 105)

    for r in results:
        d = r["model_distribution"]
        print(f"{r['threshold']:>10.3f} | {r['detection_rate']*100:>7.1f}% | "
              f"{r['savings_pct']:>7.1f}% | {r['avg_energy_mj']:>9.1f} | "
              f"{r['mean_miss_rate']:>8.4f} | "
              f"{d['nano']:>5.1f}% {d['small']:>5.1f}% {d['medium']:>5.1f}%")

    print("-" * 105)
    print("Reference baselines:")
    for name, r in [("AlwaysNano", ref_nano), ("BestFixed(small)", ref_small),
                    ("AlwaysMedium", ref_medium)]:
        d = r["model_distribution"]
        print(f"  {name:20s} | det={r['detection_rate']*100:.1f}% | "
              f"savings={r['savings_pct']:.1f}% | miss={r['mean_miss_rate']:.4f}")

    print(f"\nOracle frontier (cheapest model with true miss rate < threshold):")
    for r in oracle_results[::3]:
        d = r["model_distribution"]
        print(f"  thresh={r['threshold']:.2f} | det={r['detection_rate']*100:.1f}% | "
              f"savings={r['savings_pct']:.1f}% | "
              f"nano={d['nano']:.1f}% small={d['small']:.1f}% med={d['medium']:.1f}%")

    print(f"{'='*105}")

    # ── Plot ──
    plot_path = SCRIPT_DIR / "accuracy_vs_energy.png"
    plot_accuracy_vs_energy(results, ref_points, plot_path)


if __name__ == "__main__":
    main()
