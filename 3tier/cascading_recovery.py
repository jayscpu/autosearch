#!/usr/bin/env python3
"""
Recovery-based cascading controller + comparison with absolute controller.

Controller logic (recovery-based):
  1. Start with nano
  2. If pred_nano > max_acceptable:
     - Compute implied small miss rate: pred_nano - pred_recovery_ns
     - If implied_small < max_acceptable: use small
     - Else: use medium

Comparison plot overlays both approaches on a single accuracy-vs-energy chart.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
from controller_autosearch.models import MODELS, energy_per_window

E_NANO = energy_per_window(0)
E_SMALL = energy_per_window(1)
E_MEDIUM = energy_per_window(2)
ENERGIES = np.array([E_NANO, E_SMALL, E_MEDIUM])
MODEL_NAMES = ["nano", "small", "medium"]

TRAIN_FRAC = 0.60


# ═══════════════════════════════════════════════════════════════════
# SELECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def recovery_select(pred_nano, pred_recovery_ns, pred_recovery_sm, threshold):
    """Recovery-based cascading: start with nano, upgrade if needed."""
    n = len(pred_nano)
    selections = np.zeros(n, dtype=int)  # default: nano

    needs_upgrade = pred_nano >= threshold
    implied_small = pred_nano - pred_recovery_ns
    small_ok = implied_small < threshold

    # Upgrade to small where nano fails but implied small is OK
    selections[needs_upgrade & small_ok] = 1
    # Upgrade to medium where nano fails and small won't help enough
    selections[needs_upgrade & ~small_ok] = 2

    return selections


def absolute_select(pred_nano, pred_small, pred_medium, threshold):
    """Absolute cascading (from cascading_controller.py)."""
    n = len(pred_nano)
    selections = np.full(n, 2, dtype=int)
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
    """Compute metrics for a set of model selections."""
    n = len(selections)
    true_rates = np.array([true_nano, true_small, true_medium])
    selected_miss_rates = true_rates[selections, np.arange(n)]
    detection_rate = 1.0 - selected_miss_rates.mean()

    selected_energies = ENERGIES[selections]
    avg_energy = selected_energies.mean()
    savings_vs_medium = (1.0 - avg_energy / E_MEDIUM) * 100.0

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


# ═══════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════

def sweep_recovery(pred_nano, pred_rec_ns, pred_rec_sm,
                   true_nano, true_small, true_medium, thresholds):
    results = []
    for thresh in thresholds:
        sels = recovery_select(pred_nano, pred_rec_ns, pred_rec_sm, thresh)
        m = evaluate_selections(sels, true_nano, true_small, true_medium)
        m["threshold"] = float(thresh)
        results.append(m)
    return results


def sweep_absolute(pred_nano, pred_small, pred_medium,
                   true_nano, true_small, true_medium, thresholds):
    results = []
    for thresh in thresholds:
        sels = absolute_select(pred_nano, pred_small, pred_medium, thresh)
        m = evaluate_selections(sels, true_nano, true_small, true_medium)
        m["threshold"] = float(thresh)
        results.append(m)
    return results


# ═══════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════

def plot_comparison(recovery_results, absolute_results, oracle_results,
                    ref_points, save_path):
    """Overlay both controller curves + oracle on one plot."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Oracle frontier
    ox = [r["savings_pct"] for r in oracle_results]
    oy = [r["detection_rate"] * 100 for r in oracle_results]
    ax.plot(ox, oy, "--", color="#9C27B0", linewidth=1.5, alpha=0.5,
            label="Oracle Frontier", zorder=3)

    # Absolute controller curve
    ax_x = [r["savings_pct"] for r in absolute_results]
    ax_y = [r["detection_rate"] * 100 for r in absolute_results]
    ax.plot(ax_x, ax_y, "s-", color="#FF9800", linewidth=2, markersize=5,
            alpha=0.8, label="Absolute (3-tier)", zorder=4)

    # Recovery controller curve
    rx = [r["savings_pct"] for r in recovery_results]
    ry = [r["detection_rate"] * 100 for r in recovery_results]
    ax.plot(rx, ry, "o-", color="#2196F3", linewidth=2, markersize=6,
            label="Recovery (delta)", zorder=5)

    # Annotate recovery thresholds
    for i, r in enumerate(recovery_results):
        if i % 4 == 0 or i == len(recovery_results) - 1:
            ax.annotate(f'{r["threshold"]:.2f}',
                        (r["savings_pct"], r["detection_rate"] * 100),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color="#1565C0")

    # Reference points
    markers = {"AlwaysNano": ("^", "#F44336"), "BestFixed(small)": ("s", "#4CAF50"),
               "AlwaysMedium": ("D", "#607D8B")}
    for name, vals in ref_points.items():
        if name not in markers:
            continue
        mk, col = markers[name]
        ax.scatter(vals["savings_pct"], vals["detection_rate"] * 100,
                   marker=mk, s=150, color=col, edgecolors="black",
                   linewidths=1, zorder=10, label=name)

    ax.set_xlabel("Energy Savings vs AlwaysMedium (%)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Cascading Controller Comparison: Absolute vs Recovery",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 70)
    ax.set_ylim(40, 102)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", type=str,
                        default=str(SCRIPT_DIR / "predictions_recovery.csv"))
    parser.add_argument("--absolute", type=str,
                        default=str(SCRIPT_DIR / "predictions_3tier.csv"))
    args = parser.parse_args()

    # ── Load recovery predictions ──
    df_rec = pd.read_csv(args.recovery)
    n = len(df_rec)
    split_idx = int(n * TRAIN_FRAC)
    test_rec = df_rec.iloc[split_idx:]

    pred_nano_r = test_rec["pred_nano"].values
    pred_rec_ns = test_rec["pred_recovery_ns"].values
    pred_rec_sm = test_rec["pred_recovery_sm"].values
    true_nano_r = test_rec["true_nano"].values
    true_rec_ns = test_rec["true_recovery_ns"].values
    true_rec_sm = test_rec["true_recovery_sm"].values
    # Reconstruct true absolute miss rates from recovery predictions
    true_small_r = true_nano_r - true_rec_ns
    true_medium_r = true_small_r - true_rec_sm

    n_test = len(test_rec)
    print(f"  Recovery test split: {n_test} windows", flush=True)

    # ── Load absolute predictions ──
    df_abs = pd.read_csv(args.absolute)
    n_abs = len(df_abs)
    split_abs = int(n_abs * TRAIN_FRAC)
    test_abs = df_abs.iloc[split_abs:]

    pred_nano_a = test_abs["pred_nano"].values
    pred_small_a = test_abs["pred_small"].values
    pred_medium_a = test_abs["pred_medium"].values
    true_nano_a = test_abs["true_nano"].values
    true_small_a = test_abs["true_small"].values
    true_medium_a = test_abs["true_medium"].values

    print(f"  Absolute test split: {len(test_abs)} windows", flush=True)

    # ── Sweep both controllers ──
    thresholds = np.linspace(0.05, 0.50, 20)

    recovery_results = sweep_recovery(
        pred_nano_r, pred_rec_ns, pred_rec_sm,
        true_nano_r, true_small_r, true_medium_r, thresholds)

    absolute_results = sweep_absolute(
        pred_nano_a, pred_small_a, pred_medium_a,
        true_nano_a, true_small_a, true_medium_a, thresholds)

    oracle_results = []
    for thresh in thresholds:
        sels = oracle_select(true_nano_a, true_small_a, true_medium_a, thresh)
        m = evaluate_selections(sels, true_nano_a, true_small_a, true_medium_a)
        m["threshold"] = float(thresh)
        oracle_results.append(m)

    # ── Reference points ──
    ref_nano = evaluate_selections(
        np.zeros(n_test, dtype=int), true_nano_r, true_small_r, true_medium_r)
    ref_small = evaluate_selections(
        np.ones(n_test, dtype=int), true_nano_r, true_small_r, true_medium_r)
    ref_medium = evaluate_selections(
        np.full(n_test, 2, dtype=int), true_nano_r, true_small_r, true_medium_r)

    ref_points = {
        "AlwaysNano": ref_nano,
        "BestFixed(small)": ref_small,
        "AlwaysMedium": ref_medium,
    }

    # ── Print recovery results ──
    print(f"\n{'='*110}")
    print(f"  RECOVERY Controller Results (test split, n={n_test})")
    print(f"{'='*110}")
    print(f"{'Threshold':>10s} | {'DetRate%':>8s} | {'Savings%':>8s} | "
          f"{'AvgEnergy':>10s} | {'MissRate':>8s} | "
          f"{'Nano%':>6s} {'Small%':>6s} {'Med%':>6s}")
    print("-" * 110)

    for r in recovery_results:
        d = r["model_distribution"]
        print(f"{r['threshold']:>10.3f} | {r['detection_rate']*100:>7.1f}% | "
              f"{r['savings_pct']:>7.1f}% | {r['avg_energy_mj']:>9.1f} | "
              f"{r['mean_miss_rate']:>8.4f} | "
              f"{d['nano']:>5.1f}% {d['small']:>5.1f}% {d['medium']:>5.1f}%")

    # ── Side-by-side comparison at key thresholds ──
    print(f"\n{'='*110}")
    print(f"  HEAD-TO-HEAD COMPARISON (same threshold)")
    print(f"{'='*110}")
    print(f"{'Thresh':>7s} | {'Recovery':>28s} | {'Absolute':>28s} | {'Oracle':>28s}")
    print(f"{'':>7s} | {'det%':>8s} {'sav%':>8s} {'N/S/M':>10s} | "
          f"{'det%':>8s} {'sav%':>8s} {'N/S/M':>10s} | "
          f"{'det%':>8s} {'sav%':>8s} {'N/S/M':>10s}")
    print("-" * 110)

    for i in range(len(thresholds)):
        rr = recovery_results[i]
        ar = absolute_results[i]
        orc = oracle_results[i]
        rd, ad, od = rr["model_distribution"], ar["model_distribution"], orc["model_distribution"]
        print(f"{thresholds[i]:>7.3f} | "
              f"{rr['detection_rate']*100:>7.1f}% {rr['savings_pct']:>7.1f}% "
              f"{rd['nano']:>3.0f}/{rd['small']:>3.0f}/{rd['medium']:>3.0f} | "
              f"{ar['detection_rate']*100:>7.1f}% {ar['savings_pct']:>7.1f}% "
              f"{ad['nano']:>3.0f}/{ad['small']:>3.0f}/{ad['medium']:>3.0f} | "
              f"{orc['detection_rate']*100:>7.1f}% {orc['savings_pct']:>7.1f}% "
              f"{od['nano']:>3.0f}/{od['small']:>3.0f}/{od['medium']:>3.0f}")

    print(f"\nBaselines:")
    for name, r in [("AlwaysNano", ref_nano), ("BestFixed(small)", ref_small),
                    ("AlwaysMedium", ref_medium)]:
        print(f"  {name:20s} | det={r['detection_rate']*100:.1f}% | "
              f"savings={r['savings_pct']:.1f}%")
    print(f"{'='*110}")

    # ── Plot comparison ──
    plot_path = SCRIPT_DIR / "comparison_abs_vs_recovery.png"
    plot_comparison(recovery_results, absolute_results, oracle_results,
                    ref_points, plot_path)


if __name__ == "__main__":
    main()
