#!/usr/bin/env python3
"""
Controller simulation with the final validated predictor (3-seed average).

Compares: AlwaysNano, AlwaysMedium, BestFixed(small), Oracle,
CascadingThreshold, and BayesMPC across 5 adequacy thresholds.
"""

import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent / "final_validation"

# Hardware constants (15 frames per window)
E_NANO = 15 * 85.36     # 1280.4 mJ
E_SMALL = 15 * 128.65   # 1929.75 mJ
E_MEDIUM = 15 * 248.46  # 3726.9 mJ
ENERGY = {0: E_NANO, 1: E_SMALL, 2: E_MEDIUM}

THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35]


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_averaged_predictions():
    """Load predictions from 3 seeds, average preds, verify true values match."""
    seeds = [42, 43, 44]
    dfs = []
    for s in seeds:
        path = FINAL_DIR / f"predictions_final_seed{s}.csv"
        df = pd.read_csv(path)
        dfs.append(df)

    # Verify true values are identical across seeds
    for col in ["true_nano", "true_recovery_ns", "true_recovery_sm"]:
        vals = [df[col].values for df in dfs]
        for i in range(1, len(vals)):
            assert np.allclose(vals[0], vals[i], atol=1e-8), \
                f"True values differ across seeds for {col}"
    print("  True values verified identical across 3 seeds.", flush=True)

    # Average predictions across seeds
    base = dfs[0][["frame_idx", "split", "intersection",
                    "true_nano", "true_recovery_ns", "true_recovery_sm"]].copy()

    for pred_col in ["pred_nano", "pred_recovery_ns", "pred_recovery_sm"]:
        base[pred_col] = np.mean([df[pred_col].values for df in dfs], axis=0)

    # Compute true miss rates for all tiers
    base["true_small"] = base["true_nano"] - base["true_recovery_ns"]
    base["true_medium"] = base["true_small"] - base["true_recovery_sm"]

    # Compute predicted miss rates for all tiers
    base["pred_small"] = base["pred_nano"] - base["pred_recovery_ns"]
    base["pred_medium"] = base["pred_small"] - base["pred_recovery_sm"]

    print(f"  Total windows: {len(base)}", flush=True)
    print(f"  Splits: {base['split'].value_counts().to_dict()}", flush=True)
    return base


def train_test_split(df):
    """60/40 temporal split per intersection, maintaining order."""
    train_parts, test_parts = [], []
    for intx in df["intersection"].unique():
        sub = df[df["intersection"] == intx].sort_values("frame_idx")
        n = len(sub)
        split_idx = int(n * 0.60)
        train_parts.append(sub.iloc[:split_idx])
        test_parts.append(sub.iloc[split_idx:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    print(f"  Train: {len(train)}, Test: {len(test)}", flush=True)
    return train, test


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def true_miss_rate_for_model(row, model_idx):
    """Get the true miss rate for a given model tier."""
    if model_idx == 0:
        return row["true_nano"]
    elif model_idx == 1:
        return row["true_small"]
    else:
        return row["true_medium"]


def evaluate(df, selections, threshold):
    """Compute metrics given model selections for each window."""
    n = len(df)
    assert len(selections) == n

    energies = np.array([ENERGY[s] for s in selections])
    true_miss = np.array([true_miss_rate_for_model(df.iloc[i], selections[i])
                          for i in range(n)])
    adequate = (true_miss <= threshold).astype(float)

    switches = sum(1 for i in range(1, n) if selections[i] != selections[i - 1])

    sel_arr = np.array(selections)
    pct_nano = (sel_arr == 0).mean() * 100
    pct_small = (sel_arr == 1).mean() * 100
    pct_medium = (sel_arr == 2).mean() * 100

    return {
        "energy_savings_pct": round((1 - energies.mean() / E_MEDIUM) * 100, 2),
        "adequate_rate": round(adequate.mean() * 100, 2),
        "mean_miss_rate": round(true_miss.mean(), 6),
        "pct_nano": round(pct_nano, 1),
        "pct_small": round(pct_small, 1),
        "pct_medium": round(pct_medium, 1),
        "switches_per_100": round(switches / n * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# CONTROLLERS
# ═══════════════════════════════════════════════════════════════════

def always_nano(df, threshold):
    return [0] * len(df)


def always_medium(df, threshold):
    return [2] * len(df)


def best_fixed_small(df, threshold):
    return [1] * len(df)


def oracle(df, threshold):
    """Cheapest model whose TRUE miss rate <= threshold."""
    selections = []
    for _, row in df.iterrows():
        if row["true_nano"] <= threshold:
            selections.append(0)
        elif row["true_small"] <= threshold:
            selections.append(1)
        else:
            selections.append(2)
    return selections


def cascading_threshold(df, tau):
    """Start nano; upgrade if predicted recovery > tau."""
    selections = []
    for _, row in df.iterrows():
        model = 0  # start with nano
        if row["pred_recovery_ns"] > tau:
            model = 1  # upgrade to small
        if row["pred_recovery_sm"] > tau:
            model = 2  # upgrade to medium
        selections.append(model)
    return selections


def optimize_cascading(train_df, threshold):
    """Sweep tau on train split, pick best by objective."""
    taus = np.linspace(0.01, 0.30, 30)
    best_score, best_tau = -np.inf, 0.10
    for tau in taus:
        sels = cascading_threshold(train_df, tau)
        metrics = evaluate(train_df, sels, threshold)
        score = metrics["adequate_rate"] / 100 - 0.5 * (1 - metrics["energy_savings_pct"] / 100)
        if score > best_score:
            best_score = score
            best_tau = tau
    return best_tau


def bayes_mpc(df, H, lambda_u, lambda_o, w_s, threshold):
    """BayesMPC controller with recovery-based miss rate estimation."""
    # Precompute all 3^H sequences
    seqs = []
    for combo in itertools.product(range(3), repeat=H):
        seqs.append(list(combo))

    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}
    selections = []
    current_model = 1  # start with small
    pred_history = []

    rows = df.to_dict("records")
    for i, row in enumerate(rows):
        pred_nano = row["pred_nano"]
        pred_small = row["pred_small"]
        pred_medium = row["pred_medium"]
        pred_miss = {0: pred_nano, 1: pred_small, 2: pred_medium}

        pred_history.append(pred_nano)

        # Extrapolate future predictions (simple: repeat current)
        future_preds = []
        for h in range(H):
            if h == 0:
                future_preds.append(pred_miss.copy())
            else:
                # Linear extrapolation from recent history
                n_hist = min(len(pred_history), 3)
                recent = pred_history[-n_hist:]
                if n_hist >= 2:
                    slope = (recent[-1] - recent[0]) / (n_hist - 1)
                else:
                    slope = 0
                fut_nano = np.clip(pred_nano + slope * (h), 0, 1)
                # Keep recovery ratios stable
                fut_small = fut_nano - row["pred_recovery_ns"]
                fut_medium = fut_small - row["pred_recovery_sm"]
                future_preds.append({0: fut_nano, 1: fut_small, 2: fut_medium})

        # Evaluate all 3^H sequences
        best_cost = float("inf")
        best_first = current_model
        for seq in seqs:
            cost = 0.0
            prev = current_model
            for step in range(H):
                m = seq[step]
                # Energy cost
                cost += e_norm[m]
                # Underprediction penalty: model miss rate exceeds threshold
                miss = future_preds[step][m]
                cost += lambda_u * max(0, miss - threshold)
                # Overprediction penalty (waste): could have used cheaper model
                cost += lambda_o * max(0, threshold - miss) * 0  # disabled per spec
                # Switching cost
                if m != prev:
                    cost += w_s
                prev = m
            if cost < best_cost:
                best_cost = cost
                best_first = seq[0]

        current_model = best_first
        selections.append(best_first)

    return selections


def optimize_bayes_mpc(train_df, threshold):
    """Grid search BayesMPC hyperparameters on train split."""
    H_vals = [2, 3]
    lu_vals = [1.0, 3.0, 5.0]
    lo_vals = [0.1, 0.5]
    ws_vals = [0.01, 0.03]

    best_score = -np.inf
    best_config = {"H": 2, "lambda_u": 3.0, "lambda_o": 0.1, "w_s": 0.01}
    n_configs = len(H_vals) * len(lu_vals) * len(lo_vals) * len(ws_vals)
    print(f"    Grid search: {n_configs} configs at threshold={threshold}", flush=True)

    for H in H_vals:
        for lu in lu_vals:
            for lo in lo_vals:
                for ws in ws_vals:
                    sels = bayes_mpc(train_df, H, lu, lo, ws, threshold)
                    metrics = evaluate(train_df, sels, threshold)
                    savings = metrics["energy_savings_pct"]
                    adequacy = metrics["adequate_rate"]
                    # Pick config with best adequacy that has >20% savings
                    if savings > 20 and adequacy > best_score:
                        best_score = adequacy
                        best_config = {"H": H, "lambda_u": lu,
                                       "lambda_o": lo, "w_s": ws}

    # If no config achieved >20% savings, pick best adequacy regardless
    if best_score == -np.inf:
        print("    WARNING: no config achieved >20% savings, relaxing constraint",
              flush=True)
        for H in H_vals:
            for lu in lu_vals:
                for lo in lo_vals:
                    for ws in ws_vals:
                        sels = bayes_mpc(train_df, H, lu, lo, ws, threshold)
                        metrics = evaluate(train_df, sels, threshold)
                        score = metrics["adequate_rate"]
                        if score > best_score:
                            best_score = score
                            best_config = {"H": H, "lambda_u": lu,
                                           "lambda_o": lo, "w_s": ws}

    return best_config


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  CONTROLLER SIMULATION (final validated predictor, 3-seed avg)")
    print("=" * 70)

    df = load_averaged_predictions()

    # Split within-camera and cross-camera
    within = df[df["split"] == "within"].copy()
    cross = df[df["split"] == "cross"].copy()
    all_data = pd.concat([within, cross], ignore_index=True)

    print(f"\n  Within: {len(within)}, Cross: {len(cross)}, All: {len(all_data)}")

    train_df, test_df = train_test_split(all_data)

    all_results = []

    for threshold in THRESHOLDS:
        print(f"\n{'─'*70}")
        print(f"  THRESHOLD = {threshold}")
        print(f"{'─'*70}")

        # Baselines
        for ctrl_name, ctrl_fn in [("AlwaysNano", always_nano),
                                    ("AlwaysMedium", always_medium),
                                    ("BestFixed(small)", best_fixed_small)]:
            sels = ctrl_fn(test_df, threshold)
            metrics = evaluate(test_df, sels, threshold)
            metrics["threshold"] = threshold
            metrics["controller"] = ctrl_name
            all_results.append(metrics)
            print(f"    {ctrl_name:25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
                  f"adequate={metrics['adequate_rate']:5.1f}%", flush=True)

        # Oracle
        sels = oracle(test_df, threshold)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = "Oracle"
        all_results.append(metrics)
        print(f"    {'Oracle':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%", flush=True)

        # CascadingThreshold — optimize on train, evaluate on test
        print(f"    Optimizing CascadingThreshold...", flush=True)
        best_tau = optimize_cascading(train_df, threshold)
        sels = cascading_threshold(test_df, best_tau)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = f"CascadingThreshold(tau={best_tau:.3f})"
        all_results.append(metrics)
        print(f"    {'CascadingThreshold':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%  tau={best_tau:.3f}", flush=True)

        # BayesMPC — optimize on train, evaluate on test
        print(f"    Optimizing BayesMPC...", flush=True)
        best_config = optimize_bayes_mpc(train_df, threshold)
        sels = bayes_mpc(test_df, **best_config, threshold=threshold)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = (f"BayesMPC(H={best_config['H']},"
                                 f"lu={best_config['lambda_u']},"
                                 f"lo={best_config['lambda_o']},"
                                 f"ws={best_config['w_s']})")
        all_results.append(metrics)
        print(f"    {'BayesMPC':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%  config={best_config}",
              flush=True)

    # Save results TSV
    cols = ["threshold", "controller", "energy_savings_pct", "adequate_rate",
            "mean_miss_rate", "pct_nano", "pct_small", "pct_medium", "switches_per_100"]
    results_df = pd.DataFrame(all_results)[cols]
    tsv_path = SCRIPT_DIR / "results.tsv"
    results_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved {tsv_path}", flush=True)

    # ── Pareto plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect curves
    def get_curve(name_prefix):
        xs, ys = [], []
        for r in all_results:
            if r["controller"].startswith(name_prefix):
                xs.append(r["energy_savings_pct"])
                ys.append(r["adequate_rate"])
        return xs, ys

    # Oracle frontier
    ox, oy = get_curve("Oracle")
    ax.plot(ox, oy, "k-o", linewidth=2, markersize=8, label="Oracle", zorder=5)

    # CascadingThreshold curve
    cx, cy = get_curve("CascadingThreshold")
    ax.plot(cx, cy, "b-s", linewidth=2, markersize=7, label="CascadingThreshold", zorder=4)

    # BayesMPC curve
    bx, by = get_curve("BayesMPC")
    ax.plot(bx, by, "r-^", linewidth=2, markersize=7, label="BayesMPC", zorder=4)

    # Baseline points (use threshold=0.25 as representative)
    for r in all_results:
        if r["threshold"] == 0.25:
            if r["controller"] == "AlwaysNano":
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                          c="green", s=120, marker="D", zorder=6, label="AlwaysNano")
            elif r["controller"] == "AlwaysMedium":
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                          c="purple", s=120, marker="D", zorder=6, label="AlwaysMedium")
            elif r["controller"] == "BestFixed(small)":
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                          c="orange", s=120, marker="D", zorder=6, label="BestFixed(small)")

    # Annotate threshold values on curves
    for xs, ys, thresholds_list in [(ox, oy, THRESHOLDS), (cx, cy, THRESHOLDS),
                                     (bx, by, THRESHOLDS)]:
        for x, y, t in zip(xs, ys, thresholds_list):
            ax.annotate(f"{t}", (x, y), textcoords="offset points",
                       xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel("Energy Savings (%)", fontsize=12)
    ax.set_ylabel("Adequate Rate (%)", fontsize=12)
    ax.set_title("Controller Pareto Frontier: Energy Savings vs Adequacy", fontsize=13)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 70)
    ax.set_ylim(40, 105)

    plt.tight_layout()
    plot_path = SCRIPT_DIR / "pareto.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved {plot_path}", flush=True)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY (at threshold=0.25)")
    print(f"{'='*70}")
    for r in all_results:
        if r["threshold"] == 0.25:
            print(f"    {r['controller']:40s}: {r['energy_savings_pct']:5.1f}% savings, "
                  f"{r['adequate_rate']:5.1f}% adequacy", flush=True)

    # Final comparison line
    t25 = [r for r in all_results if r["threshold"] == 0.25]
    oracle_r = next(r for r in t25 if r["controller"] == "Oracle")
    cascade_r = next(r for r in t25 if r["controller"].startswith("CascadingThreshold"))
    bayes_r = next(r for r in t25 if r["controller"].startswith("BayesMPC"))
    print(f"\n  BayesMPC achieves {bayes_r['energy_savings_pct']:.1f}% savings at "
          f"{bayes_r['adequate_rate']:.1f}% adequacy vs "
          f"CascadingThreshold {cascade_r['energy_savings_pct']:.1f}% savings at "
          f"{cascade_r['adequate_rate']:.1f}% adequacy vs "
          f"Oracle {oracle_r['energy_savings_pct']:.1f}% at "
          f"{oracle_r['adequate_rate']:.1f}%", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
