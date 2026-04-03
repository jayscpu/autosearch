#!/usr/bin/env python3
"""
Direct recovery-based cascading controller + comparison with absolute controller.

Controller logic — start with nano (cheapest), upgrade only if recovery
justifies the energy cost:
  - if pred_recovery_ns > min_recovery: upgrade to small
  - if pred_recovery_sm > min_recovery: upgrade further to medium
These are independent checks: a window could get small only, medium only, or both.

Comparison plot overlays recovery, absolute, and oracle on one chart.
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

def direct_select(pred_recovery_ns, pred_recovery_sm, min_recovery):
    """Direct recovery-based cascading: start nano, upgrade if recovery > threshold."""
    n = len(pred_recovery_ns)
    selections = np.zeros(n, dtype=int)  # default: nano
    selections[pred_recovery_ns > min_recovery] = np.maximum(
        selections[pred_recovery_ns > min_recovery], 1)
    selections[pred_recovery_sm > min_recovery] = 2
    return selections


def oracle_direct_select(true_recovery_ns, true_recovery_sm, min_recovery):
    """Oracle version using true recovery values."""
    n = len(true_recovery_ns)
    selections = np.zeros(n, dtype=int)
    selections[true_recovery_ns > min_recovery] = np.maximum(
        selections[true_recovery_ns > min_recovery], 1)
    selections[true_recovery_sm > min_recovery] = 2
    return selections


def absolute_select(pred_nano, pred_small, pred_medium, threshold):
    """Absolute cascading (from cascading_controller.py)."""
    n = len(pred_nano)
    selections = np.full(n, 2, dtype=int)
    selections[pred_small < threshold] = 1
    selections[pred_nano < threshold] = 0
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

    # Switches per 100 windows
    switches = float(np.sum(selections[1:] != selections[:-1]))
    switches_per_100 = switches / max(n - 1, 1) * 100.0

    return {
        "detection_rate": detection_rate,
        "avg_energy_mj": avg_energy,
        "savings_pct": savings_vs_medium,
        "mean_miss_rate": float(selected_miss_rates.mean()),
        "model_distribution": dist,
        "switches_per_100": switches_per_100,
    }


# ═══════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════

def sweep_direct(pred_rec_ns, pred_rec_sm,
                 true_nano, true_small, true_medium, thresholds):
    results = []
    for thresh in thresholds:
        sels = direct_select(pred_rec_ns, pred_rec_sm, thresh)
        m = evaluate_selections(sels, true_nano, true_small, true_medium)
        m["threshold"] = float(thresh)
        results.append(m)
    return results


def sweep_oracle(true_rec_ns, true_rec_sm,
                 true_nano, true_small, true_medium, thresholds):
    results = []
    for thresh in thresholds:
        sels = oracle_direct_select(true_rec_ns, true_rec_sm, thresh)
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

def plot_comparison(direct_results, absolute_results, oracle_results,
                    ref_points, save_path):
    """Overlay direct recovery, absolute, and oracle on one plot."""
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

    # Direct recovery controller curve
    rx = [r["savings_pct"] for r in direct_results]
    ry = [r["detection_rate"] * 100 for r in direct_results]
    ax.plot(rx, ry, "o-", color="#2196F3", linewidth=2, markersize=6,
            label="Recovery (direct)", zorder=5)

    # Annotate direct recovery thresholds
    for i, r in enumerate(direct_results):
        if i % 4 == 0 or i == len(direct_results) - 1:
            ax.annotate(f'{r["threshold"]:.2f}',
                        (r["savings_pct"], r["detection_rate"] * 100),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color="#1565C0")

    # Reference points
    markers = {"AlwaysNano": ("^", "#F44336"), "BestFixed(small)": ("s", "#4CAF50"),
               "AlwaysMedium": ("D", "#607D8B")}
    for name, vals in ref_points.items():
        mk, col = markers[name]
        ax.scatter(vals["savings_pct"], vals["detection_rate"] * 100,
                   marker=mk, s=150, color=col, edgecolors="black",
                   linewidths=1, zorder=10, label=name)

    ax.set_xlabel("Energy Savings vs AlwaysMedium (%)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Cascading Controller: Direct Recovery vs Absolute",
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

    pred_rec_ns = test_rec["pred_recovery_ns"].values
    pred_rec_sm = test_rec["pred_recovery_sm"].values
    true_nano_r = test_rec["true_nano"].values
    true_rec_ns = test_rec["true_recovery_ns"].values
    true_rec_sm = test_rec["true_recovery_sm"].values
    # Reconstruct true absolute miss rates
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

    # ── Sweep all controllers ──
    thresholds = np.linspace(0.01, 0.30, 20)

    direct_results = sweep_direct(
        pred_rec_ns, pred_rec_sm,
        true_nano_r, true_small_r, true_medium_r, thresholds)

    oracle_results = sweep_oracle(
        true_rec_ns, true_rec_sm,
        true_nano_r, true_small_r, true_medium_r, thresholds)

    # Absolute uses different threshold range (0.05-0.50) for fair comparison
    abs_thresholds = np.linspace(0.05, 0.50, 20)
    absolute_results = sweep_absolute(
        pred_nano_a, pred_small_a, pred_medium_a,
        true_nano_a, true_small_a, true_medium_a, abs_thresholds)

    # ── Reference points (use recovery true values) ──
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

    # ── Print direct recovery results ──
    print(f"\n{'='*120}")
    print(f"  DIRECT RECOVERY Controller Results (test split, n={n_test})")
    print(f"{'='*120}")
    print(f"{'MinRecov':>10s} | {'DetRate%':>8s} | {'Savings%':>8s} | "
          f"{'AvgEnergy':>10s} | {'MissRate':>8s} | "
          f"{'Nano%':>6s} {'Small%':>6s} {'Med%':>6s} | {'Sw/100':>6s}")
    print("-" * 120)

    for r in direct_results:
        d = r["model_distribution"]
        print(f"{r['threshold']:>10.3f} | {r['detection_rate']*100:>7.1f}% | "
              f"{r['savings_pct']:>7.1f}% | {r['avg_energy_mj']:>9.1f} | "
              f"{r['mean_miss_rate']:>8.4f} | "
              f"{d['nano']:>5.1f}% {d['small']:>5.1f}% {d['medium']:>5.1f}% | "
              f"{r['switches_per_100']:>5.1f}")

    # ── Oracle results ──
    print(f"\n{'='*120}")
    print(f"  ORACLE (true recovery values)")
    print(f"{'='*120}")
    for r in oracle_results:
        d = r["model_distribution"]
        print(f"{r['threshold']:>10.3f} | {r['detection_rate']*100:>7.1f}% | "
              f"{r['savings_pct']:>7.1f}% | {r['avg_energy_mj']:>9.1f} | "
              f"{r['mean_miss_rate']:>8.4f} | "
              f"{d['nano']:>5.1f}% {d['small']:>5.1f}% {d['medium']:>5.1f}% | "
              f"{r['switches_per_100']:>5.1f}")

    # ── Head-to-head at comparable savings levels ──
    print(f"\n{'='*120}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*120}")
    print(f"{'':>12s} | {'Direct Recovery':>35s} | {'Absolute (3-tier)':>35s} | {'Oracle':>35s}")
    print(f"{'MinRec/Thr':>12s} | {'det%':>8s} {'sav%':>8s} {'N/S/M':>10s} {'sw':>5s} | "
          f"{'det%':>8s} {'sav%':>8s} {'N/S/M':>10s} {'sw':>5s} | "
          f"{'det%':>8s} {'sav%':>8s} {'N/S/M':>10s} {'sw':>5s}")
    print("-" * 120)

    for i in range(len(thresholds)):
        rr = direct_results[i]
        orc = oracle_results[i]
        rd, od = rr["model_distribution"], orc["model_distribution"]
        # Find closest absolute result by savings
        j = min(range(len(absolute_results)),
                key=lambda k: abs(absolute_results[k]["savings_pct"] - rr["savings_pct"]))
        ar = absolute_results[j]
        ad = ar["model_distribution"]
        print(f"{thresholds[i]:>12.3f} | "
              f"{rr['detection_rate']*100:>7.1f}% {rr['savings_pct']:>7.1f}% "
              f"{rd['nano']:>3.0f}/{rd['small']:>3.0f}/{rd['medium']:>3.0f} {rr['switches_per_100']:>4.1f} | "
              f"{ar['detection_rate']*100:>7.1f}% {ar['savings_pct']:>7.1f}% "
              f"{ad['nano']:>3.0f}/{ad['small']:>3.0f}/{ad['medium']:>3.0f} {ar['switches_per_100']:>4.1f} | "
              f"{orc['detection_rate']*100:>7.1f}% {orc['savings_pct']:>7.1f}% "
              f"{od['nano']:>3.0f}/{od['small']:>3.0f}/{od['medium']:>3.0f} {orc['switches_per_100']:>4.1f}")

    print(f"\nBaselines:")
    for name, r in [("AlwaysNano", ref_nano), ("BestFixed(small)", ref_small),
                    ("AlwaysMedium", ref_medium)]:
        print(f"  {name:20s} | det={r['detection_rate']*100:.1f}% | "
              f"savings={r['savings_pct']:.1f}% | sw/100={r['switches_per_100']:.1f}")
    print(f"{'='*120}")

    # ── Plot comparison ──
    plot_path = SCRIPT_DIR / "accuracy_vs_energy_comparison.png"
    plot_comparison(direct_results, absolute_results, oracle_results,
                    ref_points, plot_path)


if __name__ == "__main__":
    main()
