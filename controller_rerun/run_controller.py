#!/usr/bin/env python3
"""
Controller simulation with corrected adequacy metric.

Adequacy is measured only on solvable windows (where at least one model
achieves miss rate < threshold). Unsolvable windows are excluded.
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

THRESHOLDS = [0.30, 0.35, 0.40, 0.50]


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════

def load_averaged_predictions():
    seeds = [42, 43, 44]
    dfs = [pd.read_csv(FINAL_DIR / f"predictions_final_seed{s}.csv") for s in seeds]

    for col in ["true_nano", "true_recovery_ns", "true_recovery_sm"]:
        vals = [df[col].values for df in dfs]
        for i in range(1, len(vals)):
            assert np.allclose(vals[0], vals[i], atol=1e-8), \
                f"True values differ across seeds for {col}"
    print("  True values verified identical across 3 seeds.", flush=True)

    base = dfs[0][["frame_idx", "split", "intersection",
                    "true_nano", "true_recovery_ns", "true_recovery_sm"]].copy()
    for pred_col in ["pred_nano", "pred_recovery_ns", "pred_recovery_sm"]:
        base[pred_col] = np.mean([df[pred_col].values for df in dfs], axis=0)

    # True miss rates for all tiers
    base["true_small"] = base["true_nano"] - base["true_recovery_ns"]
    base["true_medium"] = base["true_small"] - base["true_recovery_sm"]

    # Predicted miss rates for all tiers
    base["pred_small"] = base["pred_nano"] - base["pred_recovery_ns"]
    base["pred_medium"] = base["pred_small"] - base["pred_recovery_sm"]

    print(f"  Total windows: {len(base)}", flush=True)
    return base


def train_test_split(df):
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
# ORACLE & SOLVABILITY
# ═══════════════════════════════════════════════════════════════════

def oracle_choice(row, threshold):
    """Return cheapest adequate model, or -1 if unsolvable."""
    if row["true_nano"] < threshold:
        return 0
    elif row["true_small"] < threshold:
        return 1
    elif row["true_medium"] < threshold:
        return 2
    else:
        return -1  # unsolvable


def get_solvable_mask(df, threshold):
    """Return boolean mask of solvable windows and oracle choices."""
    oracle_choices = df.apply(lambda r: oracle_choice(r, threshold), axis=1).values
    solvable = oracle_choices >= 0
    return solvable, oracle_choices


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate(df, selections, threshold):
    """Compute metrics on solvable windows only."""
    solvable, oracle_choices = get_solvable_mask(df, threshold)
    n_total = len(df)
    n_solvable = int(solvable.sum())

    if n_solvable == 0:
        return {"n_solvable": 0, "n_total": n_total,
                "energy_savings_pct": 0, "adequate_rate": 0,
                "correct_rate": 0, "over_provision_rate": 0,
                "mean_miss_rate": 0, "pct_nano": 0, "pct_small": 0,
                "pct_medium": 0, "switches_per_100": 0}

    # Filter to solvable windows
    sel_arr = np.array(selections)
    sol_sel = sel_arr[solvable]
    sol_oracle = oracle_choices[solvable]

    # Adequacy: selected model >= oracle choice
    adequate = (sol_sel >= sol_oracle).astype(float)
    # Correct: exactly oracle's choice
    correct = (sol_sel == sol_oracle).astype(float)
    # Over-provisioned: more expensive than needed
    over_prov = (sol_sel > sol_oracle).astype(float)

    # Energy on solvable windows
    energies = np.array([ENERGY[s] for s in sol_sel])

    # True miss rate of selected model on solvable windows
    sol_df = df[solvable].reset_index(drop=True)
    true_miss = np.array([
        sol_df.iloc[i]["true_nano"] if sol_sel[i] == 0
        else sol_df.iloc[i]["true_small"] if sol_sel[i] == 1
        else sol_df.iloc[i]["true_medium"]
        for i in range(n_solvable)
    ])

    # Switches (on all windows, not just solvable)
    switches = sum(1 for i in range(1, n_total) if selections[i] != selections[i - 1])

    # Model distribution (on solvable)
    pct_nano = (sol_sel == 0).mean() * 100
    pct_small = (sol_sel == 1).mean() * 100
    pct_medium = (sol_sel == 2).mean() * 100

    return {
        "n_solvable": n_solvable,
        "n_total": n_total,
        "energy_savings_pct": round((1 - energies.mean() / E_MEDIUM) * 100, 2),
        "adequate_rate": round(adequate.mean() * 100, 2),
        "correct_rate": round(correct.mean() * 100, 2),
        "over_provision_rate": round(over_prov.mean() * 100, 2),
        "mean_miss_rate": round(float(true_miss.mean()), 6),
        "pct_nano": round(pct_nano, 1),
        "pct_small": round(pct_small, 1),
        "pct_medium": round(pct_medium, 1),
        "switches_per_100": round(switches / n_total * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# CONTROLLERS
# ═══════════════════════════════════════════════════════════════════

def always_nano(df, **kw):
    return [0] * len(df)

def always_medium(df, **kw):
    return [2] * len(df)

def best_fixed_small(df, **kw):
    return [1] * len(df)

def oracle_controller(df, threshold):
    solvable, oracle_choices = get_solvable_mask(df, threshold)
    # For unsolvable windows, oracle picks medium (best effort)
    return [int(c) if c >= 0 else 2 for c in oracle_choices]


def cascading_threshold(df, tau):
    selections = []
    for _, row in df.iterrows():
        model = 0
        if row["pred_recovery_ns"] > tau:
            model = 1
        if row["pred_recovery_sm"] > tau:
            model = 2
        selections.append(model)
    return selections


def optimize_cascading(train_df, threshold):
    taus = np.linspace(0.01, 0.30, 30)
    best_adequate, best_tau = -1.0, 0.10
    for tau in taus:
        sels = cascading_threshold(train_df, tau)
        metrics = evaluate(train_df, sels, threshold)
        if metrics["adequate_rate"] > best_adequate:
            best_adequate = metrics["adequate_rate"]
            best_tau = tau
    return best_tau


def cascading_threshold_2d(df, tau_ns, tau_sm):
    selections = []
    for _, row in df.iterrows():
        model = 0
        if row["pred_recovery_ns"] > tau_ns:
            model = 1
        if row["pred_recovery_sm"] > tau_sm:
            model = 2
        selections.append(model)
    return selections


def optimize_cascading_2d(train_df, threshold):
    taus = np.linspace(0.01, 0.30, 30)
    best_adequate, best_savings = -1.0, -999.0
    best_tau_ns, best_tau_sm = 0.10, 0.10
    for tau_ns in taus:
        for tau_sm in taus:
            sels = cascading_threshold_2d(train_df, tau_ns, tau_sm)
            metrics = evaluate(train_df, sels, threshold)
            adq = metrics["adequate_rate"]
            sav = metrics["energy_savings_pct"]
            if adq > best_adequate or (adq == best_adequate and sav > best_savings):
                best_adequate = adq
                best_savings = sav
                best_tau_ns = tau_ns
                best_tau_sm = tau_sm
    return best_tau_ns, best_tau_sm


def bayes_mpc(df, H, lambda_u, w_s, threshold):
    """MPC controller: minimize energy + under-provision penalty + switch cost."""
    # Precompute all 3^H sequences
    seqs = list(itertools.product(range(3), repeat=H))
    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}

    selections = []
    current_model = 1
    rows = df.to_dict("records")

    for i, row in enumerate(rows):
        pred_miss = {
            0: row["pred_nano"],
            1: row["pred_small"],
            2: row["pred_medium"],
        }

        # Evaluate all sequences
        best_cost = float("inf")
        best_first = current_model
        for seq in seqs:
            cost = 0.0
            prev = current_model
            for step in range(H):
                m = seq[step]
                # Energy
                cost += e_norm[m]
                # Under-provision penalty
                cost += lambda_u * max(0, pred_miss[m] - threshold)
                # Switch cost
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
    H_vals = [1, 2, 3]
    lu_vals = [1.0, 3.0, 5.0, 10.0]
    ws_vals = [0.0, 0.01, 0.03]

    n_configs = len(H_vals) * len(lu_vals) * len(ws_vals)
    print(f"    Grid search: {n_configs} configs at threshold={threshold}", flush=True)

    best_adequate = -1.0
    best_config = {"H": 1, "lambda_u": 5.0, "w_s": 0.01}

    for H in H_vals:
        for lu in lu_vals:
            for ws in ws_vals:
                sels = bayes_mpc(train_df, H, lu, ws, threshold)
                metrics = evaluate(train_df, sels, threshold)
                savings = metrics["energy_savings_pct"]
                adequacy = metrics["adequate_rate"]
                if savings > 10 and adequacy > best_adequate:
                    best_adequate = adequacy
                    best_config = {"H": H, "lambda_u": lu, "w_s": ws}

    # Fallback if nothing achieved >10% savings
    if best_adequate < 0:
        print("    WARNING: no config achieved >10% savings, relaxing", flush=True)
        for H in H_vals:
            for lu in lu_vals:
                for ws in ws_vals:
                    sels = bayes_mpc(train_df, H, lu, ws, threshold)
                    metrics = evaluate(train_df, sels, threshold)
                    if metrics["adequate_rate"] > best_adequate:
                        best_adequate = metrics["adequate_rate"]
                        best_config = {"H": H, "lambda_u": lu, "w_s": ws}

    return best_config


def rich_mpc(df, lambda_u, lambda_o, threshold):
    """RichMPC: Bayes Risk with both lambda_under and lambda_over."""
    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}
    e_rel_nano = {m: ENERGY[m] / E_NANO for m in range(3)}

    selections = []
    rows = df.to_dict("records")

    for row in rows:
        pred_miss = {
            0: row["pred_nano"],
            1: row["pred_small"],
            2: row["pred_medium"],
        }

        best_cost = float("inf")
        best_m = 2
        for m in range(3):
            cost = e_norm[m]
            cost += lambda_u * max(0, pred_miss[m] - threshold)
            cost += lambda_o * max(0, threshold - pred_miss[m]) * e_rel_nano[m]
            if cost < best_cost:
                best_cost = cost
                best_m = m

        selections.append(best_m)

    return selections


def optimize_rich_mpc(train_df, threshold):
    lu_vals = [1.0, 3.0, 5.0, 10.0, 20.0]
    lo_vals = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    n_configs = len(lu_vals) * len(lo_vals)
    print(f"    Grid search: {n_configs} configs at threshold={threshold}", flush=True)

    best_score = -1.0
    best_config = {"lambda_u": 5.0, "lambda_o": 0.5}

    for lu in lu_vals:
        for lo in lo_vals:
            sels = rich_mpc(train_df, lu, lo, threshold)
            metrics = evaluate(train_df, sels, threshold)
            # Score: prioritize adequacy but reward savings
            score = metrics["adequate_rate"] / 100 + 0.3 * metrics["energy_savings_pct"] / 100
            if score > best_score:
                best_score = score
                best_config = {"lambda_u": lu, "lambda_o": lo}

    return best_config


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  CONTROLLER SIMULATION (corrected adequacy, solvable windows only)")
    print("=" * 70)

    df = load_averaged_predictions()
    all_data = pd.concat([
        df[df["split"] == "within"],
        df[df["split"] == "cross"],
    ], ignore_index=True)
    print(f"  Total: {len(all_data)}")

    train_df, test_df = train_test_split(all_data)

    all_results = []

    for threshold in THRESHOLDS:
        print(f"\n{'─'*70}")
        print(f"  THRESHOLD = {threshold}")
        print(f"{'─'*70}")

        solvable, _ = get_solvable_mask(test_df, threshold)
        n_solvable = int(solvable.sum())
        n_total = len(test_df)
        print(f"  Solvable: {n_solvable}/{n_total} ({n_solvable/n_total*100:.1f}%)",
              flush=True)

        # Baselines
        for ctrl_name, ctrl_fn in [("AlwaysNano", always_nano),
                                    ("AlwaysMedium", always_medium),
                                    ("BestFixed(small)", best_fixed_small)]:
            sels = ctrl_fn(test_df)
            metrics = evaluate(test_df, sels, threshold)
            metrics["threshold"] = threshold
            metrics["controller"] = ctrl_name
            all_results.append(metrics)
            print(f"    {ctrl_name:25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
                  f"adequate={metrics['adequate_rate']:5.1f}%  "
                  f"correct={metrics['correct_rate']:5.1f}%", flush=True)

        # Oracle
        sels = oracle_controller(test_df, threshold)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = "Oracle"
        all_results.append(metrics)
        print(f"    {'Oracle':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%  "
              f"correct={metrics['correct_rate']:5.1f}%", flush=True)

        # CascadingThreshold
        print(f"    Optimizing CascadingThreshold...", flush=True)
        best_tau = optimize_cascading(train_df, threshold)
        sels = cascading_threshold(test_df, best_tau)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = f"CascadingThreshold(tau={best_tau:.3f})"
        all_results.append(metrics)
        print(f"    {'CascadingThreshold':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%  "
              f"correct={metrics['correct_rate']:5.1f}%  tau={best_tau:.3f}", flush=True)

        # CascadingThreshold2D
        print(f"    Optimizing CascadingThreshold2D...", flush=True)
        best_tau_ns, best_tau_sm = optimize_cascading_2d(train_df, threshold)
        sels = cascading_threshold_2d(test_df, best_tau_ns, best_tau_sm)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = (f"CascadingThreshold2D("
                                 f"ns={best_tau_ns:.3f},sm={best_tau_sm:.3f})")
        all_results.append(metrics)
        print(f"    {'CascadingThreshold2D':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%  "
              f"correct={metrics['correct_rate']:5.1f}%  "
              f"tau_ns={best_tau_ns:.3f} tau_sm={best_tau_sm:.3f}", flush=True)

        # BayesMPC
        print(f"    Optimizing BayesMPC...", flush=True)
        best_config = optimize_bayes_mpc(train_df, threshold)
        sels = bayes_mpc(test_df, **best_config, threshold=threshold)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = (f"BayesMPC(H={best_config['H']},"
                                 f"lu={best_config['lambda_u']},"
                                 f"ws={best_config['w_s']})")
        all_results.append(metrics)
        print(f"    {'BayesMPC':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%  "
              f"correct={metrics['correct_rate']:5.1f}%  config={best_config}",
              flush=True)

        # RichMPC
        print(f"    Optimizing RichMPC...", flush=True)
        best_rich = optimize_rich_mpc(train_df, threshold)
        sels = rich_mpc(test_df, **best_rich, threshold=threshold)
        metrics = evaluate(test_df, sels, threshold)
        metrics["threshold"] = threshold
        metrics["controller"] = (f"RichMPC(lu={best_rich['lambda_u']},"
                                 f"lo={best_rich['lambda_o']})")
        all_results.append(metrics)
        print(f"    {'RichMPC':25s}: savings={metrics['energy_savings_pct']:5.1f}%  "
              f"adequate={metrics['adequate_rate']:5.1f}%  "
              f"correct={metrics['correct_rate']:5.1f}%  "
              f"nano={metrics['pct_nano']:.0f}% sm={metrics['pct_small']:.0f}% "
              f"med={metrics['pct_medium']:.0f}%  config={best_rich}", flush=True)

    # Save TSV
    cols = ["threshold", "n_solvable", "n_total", "controller",
            "energy_savings_pct", "adequate_rate", "correct_rate",
            "over_provision_rate", "mean_miss_rate",
            "pct_nano", "pct_small", "pct_medium", "switches_per_100"]
    results_df = pd.DataFrame(all_results)[cols]
    tsv_path = SCRIPT_DIR / "results.tsv"
    results_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved {tsv_path}", flush=True)

    # ── Pareto plot (one subplot per threshold) ──────────────────
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(5 * len(THRESHOLDS), 5),
                             sharey=True)
    if len(THRESHOLDS) == 1:
        axes = [axes]

    for ax, threshold in zip(axes, THRESHOLDS):
        t_results = [r for r in all_results if r["threshold"] == threshold]

        # Oracle
        oracle_r = next(r for r in t_results if r["controller"] == "Oracle")
        ax.scatter(oracle_r["energy_savings_pct"], oracle_r["adequate_rate"],
                  c="black", s=150, marker="*", zorder=6, label="Oracle")

        # Baselines
        colors = {"AlwaysNano": "green", "AlwaysMedium": "purple",
                  "BestFixed(small)": "orange"}
        for r in t_results:
            if r["controller"] in colors:
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                          c=colors[r["controller"]], s=100, marker="D",
                          zorder=5, label=r["controller"])

        # CascadingThreshold
        cascade_r = next(r for r in t_results
                         if r["controller"].startswith("CascadingThreshold("))
        ax.scatter(cascade_r["energy_savings_pct"], cascade_r["adequate_rate"],
                  c="blue", s=100, marker="s", zorder=5, label="CascadingThreshold")

        # CascadingThreshold2D
        cascade2d_r = next(r for r in t_results
                           if r["controller"].startswith("CascadingThreshold2D"))
        ax.scatter(cascade2d_r["energy_savings_pct"], cascade2d_r["adequate_rate"],
                  c="cyan", s=100, marker="p", zorder=5, label="CascadingThreshold2D")

        # BayesMPC
        bayes_r = next(r for r in t_results
                       if r["controller"].startswith("BayesMPC"))
        ax.scatter(bayes_r["energy_savings_pct"], bayes_r["adequate_rate"],
                  c="red", s=100, marker="^", zorder=5, label="BayesMPC")

        # RichMPC
        rich_r = next(r for r in t_results
                      if r["controller"].startswith("RichMPC"))
        ax.scatter(rich_r["energy_savings_pct"], rich_r["adequate_rate"],
                  c="magenta", s=120, marker="v", zorder=6, label="RichMPC")

        n_solv = next(r for r in t_results)["n_solvable"]
        n_tot = next(r for r in t_results)["n_total"]
        ax.set_title(f"T={threshold} ({n_solv}/{n_tot} solvable)", fontsize=11)
        ax.set_xlabel("Energy Savings (%)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    axes[0].set_ylabel("Adequate Rate (%, solvable only)", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plot_path = SCRIPT_DIR / "pareto.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved {plot_path}", flush=True)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for threshold in THRESHOLDS:
        t_results = [r for r in all_results if r["threshold"] == threshold]
        oracle_r = next(r for r in t_results if r["controller"] == "Oracle")
        cascade_r = next(r for r in t_results
                         if r["controller"].startswith("CascadingThreshold("))
        bayes_r = next(r for r in t_results
                       if r["controller"].startswith("BayesMPC"))
        rich_r = next(r for r in t_results
                      if r["controller"].startswith("RichMPC"))
        cascade2d_r = next(r for r in t_results
                           if r["controller"].startswith("CascadingThreshold2D"))
        print(f"\n  T={threshold} ({oracle_r['n_solvable']}/{oracle_r['n_total']} solvable):")
        print(f"    Oracle:             {oracle_r['energy_savings_pct']:5.1f}% savings, "
              f"{oracle_r['adequate_rate']:5.1f}% adequate")
        print(f"    RichMPC:            {rich_r['energy_savings_pct']:5.1f}% savings, "
              f"{rich_r['adequate_rate']:5.1f}% adequate")
        print(f"    BayesMPC:           {bayes_r['energy_savings_pct']:5.1f}% savings, "
              f"{bayes_r['adequate_rate']:5.1f}% adequate")
        print(f"    CascadingThreshold: {cascade_r['energy_savings_pct']:5.1f}% savings, "
              f"{cascade_r['adequate_rate']:5.1f}% adequate")
        print(f"    CascThreshold2D:    {cascade2d_r['energy_savings_pct']:5.1f}% savings, "
              f"{cascade2d_r['adequate_rate']:5.1f}% adequate")

    # ── Comparison table ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMPARISON: BayesMPC vs RichMPC vs Oracle")
    print(f"{'='*70}")
    print(f"  {'Threshold':>9s} | {'BayesMPC savings/adq':>22s} | "
          f"{'RichMPC savings/adq':>22s} | {'Oracle savings':>14s} | "
          f"{'RichMPC nano%/sm%/med%':>22s}")
    print(f"  {'-'*95}")
    for threshold in THRESHOLDS:
        t_results = [r for r in all_results if r["threshold"] == threshold]
        oracle_r = next(r for r in t_results if r["controller"] == "Oracle")
        bayes_r = next(r for r in t_results
                       if r["controller"].startswith("BayesMPC"))
        rich_r = next(r for r in t_results
                      if r["controller"].startswith("RichMPC"))
        print(f"  {threshold:>9.2f} | "
              f"{bayes_r['energy_savings_pct']:5.1f}% / {bayes_r['adequate_rate']:5.1f}%    | "
              f"{rich_r['energy_savings_pct']:5.1f}% / {rich_r['adequate_rate']:5.1f}%    | "
              f"{oracle_r['energy_savings_pct']:5.1f}%         | "
              f"{rich_r['pct_nano']:4.0f}% / {rich_r['pct_small']:4.0f}% / {rich_r['pct_medium']:4.0f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
