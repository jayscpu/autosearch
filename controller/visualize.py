"""
Visualization functions for controller simulation results.

Generates thesis-quality plots: timeline model selections, energy comparison
bar charts, and per-intersection breakdowns.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color scheme for model tiers
MODEL_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}  # green, yellow, red
MODEL_NAMES = {0: "nano", 1: "small", 2: "medium"}


def plot_timeline(pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
                  results: dict, oracle_name: str = "Oracle",
                  threshold_name: str = None, mpc_name: str = None,
                  t1: float = 0.15, t2: float = 0.35,
                  predictor_name: str = "LSTM", save_name: str = "timeline"):
    """Timeline plot: miss rates + model selections over time.

    Top subplot: true and predicted miss rate traces with threshold lines.
    Bottom subplot: model selection bands for oracle and best controllers.
    """
    n = len(pred_miss_rates)
    x = np.arange(n)

    # Collect controller results to plot (oracle + up to 2 adaptive)
    ctrl_results = {}
    for r in results:
        ctrl_results[r["controller"]] = r

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1.2]})

    # ── Top: miss rate traces ─────────────────────────────────────────────
    ax1.plot(x, true_miss_rates, color="#1565C0", linewidth=0.8, alpha=0.7,
             label="True miss rate")
    ax1.plot(x, pred_miss_rates, color="#E65100", linewidth=0.8, alpha=0.7,
             label="Predicted miss rate")
    ax1.axhline(y=t1, color="gray", linestyle="--", linewidth=0.8,
                label=f"t1={t1:.2f}")
    ax1.axhline(y=t2, color="gray", linestyle="-.", linewidth=0.8,
                label=f"t2={t2:.2f}")
    ax1.set_ylabel("Miss Rate")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(f"{predictor_name} — Prediction vs Truth", fontsize=11)

    # ── Bottom: model selection bands ─────────────────────────────────────
    controllers_to_plot = []
    # Always include oracle if available
    for r in results:
        if "Oracle" in r["controller"]:
            controllers_to_plot.append(("Oracle", r["oracle_selections"]))
            break

    # Add best threshold
    if threshold_name:
        for r in results:
            if r["controller"] == threshold_name:
                controllers_to_plot.append(
                    (f"Threshold ({r['energy_savings_pct']:.1f}% savings)",
                     r["selections"]))
                break

    # Add best MPC
    if mpc_name:
        for r in results:
            if r["controller"] == mpc_name:
                controllers_to_plot.append(
                    (f"MPC ({r['energy_savings_pct']:.1f}% savings)",
                     r["selections"]))
                break

    # If no specific names, pick from results
    if not controllers_to_plot:
        for r in results:
            controllers_to_plot.append((r["controller"], r["selections"]))
            if len(controllers_to_plot) >= 3:
                break

    n_ctrls = len(controllers_to_plot)
    for idx, (label, sels) in enumerate(controllers_to_plot):
        y_offset = n_ctrls - 1 - idx
        for t_idx in range(n):
            color = MODEL_COLORS[sels[t_idx]]
            ax2.barh(y_offset, 1, left=t_idx, height=0.8, color=color,
                     edgecolor="none")

    ax2.set_yticks(range(n_ctrls))
    ax2.set_yticklabels([c[0] for c in reversed(controllers_to_plot)], fontsize=8)
    ax2.set_xlabel("Decision Window Index")
    ax2.set_xlim(0, n)

    # Legend for model colors
    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_NAMES[m])
               for m in [0, 1, 2]]
    ax2.legend(handles=patches, loc="upper right", fontsize=8, ncol=3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_energy_comparison(results: list, oracle_energy: float = None,
                           medium_energy: float = None,
                           save_name: str = "energy_comparison"):
    """Horizontal bar chart of energy consumption for all controllers.

    Bars colored by adequate_rate: green >=0.90, yellow >=0.80, red <0.80.
    Oracle and always-medium reference lines are marked.
    """
    # Sort by energy ascending
    sorted_results = sorted(results, key=lambda r: r["avg_energy_mj"])

    names = [r["controller"] for r in sorted_results]
    energies = [r["avg_energy_mj"] for r in sorted_results]
    adequacies = [r["adequate_rate"] for r in sorted_results]

    # Color by adequacy
    colors = []
    for a in adequacies:
        if a >= 0.90:
            colors.append("#4CAF50")  # green
        elif a >= 0.80:
            colors.append("#FFC107")  # yellow
        else:
            colors.append("#F44336")  # red

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, energies, color=colors, edgecolor="white", height=0.7)

    # Add adequacy labels on bars
    for i, (bar, adq) in enumerate(zip(bars, adequacies)):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"adq={adq:.2f}", va="center", fontsize=7, color="gray")

    # Reference lines
    if oracle_energy is not None:
        ax.axvline(x=oracle_energy, color="#1565C0", linestyle="--",
                   linewidth=1.2, label=f"Oracle ({oracle_energy:.0f} mJ)")
    if medium_energy is not None:
        ax.axvline(x=medium_energy, color="#B71C1C", linestyle="--",
                   linewidth=1.2, label=f"AlwaysMedium ({medium_energy:.0f} mJ)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Avg Energy per Window (mJ)")
    ax.set_title("Controller Energy Comparison")
    ax.legend(fontsize=8)

    # Color legend
    patches = [
        mpatches.Patch(color="#4CAF50", label="adequate ≥ 0.90"),
        mpatches.Patch(color="#FFC107", label="adequate ≥ 0.80"),
        mpatches.Patch(color="#F44336", label="adequate < 0.80"),
    ]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:2],
              loc="lower right", fontsize=7)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_per_intersection(intersection_results: dict, top_controllers: list,
                          cross_cam_name: str = "Bellevue_150th_SE38th",
                          save_name: str = "per_intersection"):
    """Grouped bar chart: energy savings per intersection for top controllers.

    Only for 4CAM datasets. Cross-camera intersection is highlighted.

    Args:
        intersection_results: {intersection_name: {controller_name: metrics_dict}}
        top_controllers: List of controller names to compare (max 3).
        cross_cam_name: Name of the cross-camera intersection to highlight.
    """
    intersections = sorted(intersection_results.keys())
    n_inter = len(intersections)
    n_ctrl = len(top_controllers)

    if n_inter == 0 or n_ctrl == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, max(5, n_inter * 0.5)))

    bar_height = 0.8 / n_ctrl
    ctrl_colors = ["#1565C0", "#E65100", "#2E7D32"]

    for c_idx, ctrl_name in enumerate(top_controllers[:3]):
        y_positions = []
        savings = []
        for i, inter in enumerate(intersections):
            y_positions.append(i + c_idx * bar_height - 0.4 + bar_height / 2)
            if ctrl_name in intersection_results[inter]:
                savings.append(
                    intersection_results[inter][ctrl_name]["energy_savings_pct"])
            else:
                savings.append(0.0)

        bars = ax.barh(y_positions, savings, height=bar_height,
                       color=ctrl_colors[c_idx], label=ctrl_name, alpha=0.85)

    # Highlight cross-camera intersection
    for i, inter in enumerate(intersections):
        if cross_cam_name in inter:
            ax.axhspan(i - 0.5, i + 0.5, color="#FFEB3B", alpha=0.15, zorder=0)
            ax.text(ax.get_xlim()[1] * 0.02, i, "★ cross-cam",
                    fontsize=7, color="#B71C1C", va="center")

    ax.set_yticks(range(n_inter))
    ax.set_yticklabels(intersections, fontsize=8)
    ax.set_xlabel("Energy Savings vs AlwaysMedium (%)")
    ax.set_title("Per-Intersection Energy Savings (4CAM)")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
