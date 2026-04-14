#!/usr/bin/env python3
"""Alternative evaluation metrics for controllers.

For each threshold, re-optimize DirectThreshold and EnergyPenalty
hyperparameters PER METRIC on the train split, then evaluate all five
metrics on the test split.

Metrics (all computed over solvable windows):
  1. Binary Adequacy (adq + 0.5*sav) — the current baseline.
  2. Efficiency-Weighted Quality (EWQ): quality * (1 + efficiency).
  3. Miss-Rate-Aware Savings (MRAS): energy_saved if acceptable else -1.
  4. Pareto: adq_rate * sav_rate.
  5. Threshold-Relative Miss Rate (TRMR).
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PRED_DIR = PROJECT_DIR / "refinedsys"

E_NANO = 15 * 85.36
E_SMALL = 15 * 128.65
E_MEDIUM = 15 * 248.46
ENERGY = np.array([E_NANO, E_SMALL, E_MEDIUM])
THRESHOLDS = [0.30, 0.35, 0.40, 0.50]
N_STEPS = 5
SEEDS = [42, 43, 44]
MARGINS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
LU_GRID = [1.0, 3.0, 5.0, 10.0, 20.0]
MRAS_PENALTY = 1.0

METRICS = ["binary", "ewq", "mras", "pareto", "trmr"]


# ── Data ────────────────────────────────────────────────────────────

def load_predictions():
    dfs = [pd.read_csv(PRED_DIR / f"predictions_multistep_seed{s}.csv")
           for s in SEEDS]
    true_cols = [c for c in dfs[0].columns if c.startswith("true_")]
    for c in true_cols:
        v = [df[c].values for df in dfs]
        for i in range(1, len(v)):
            assert np.allclose(v[0], v[i], atol=1e-8), f"true differ: {c}"
    print("  True identical across 3 seeds.", flush=True)
    base = dfs[0][["frame_idx", "split", "intersection"]].copy()
    for c in [c for c in dfs[0].columns if c.startswith("pred_")]:
        base[c] = np.mean([df[c].values for df in dfs], axis=0)
    for c in true_cols:
        base[c] = dfs[0][c].values
    # Per-tier miss rates per step for pred_small_avg / pred_medium_avg.
    for s in range(N_STEPS):
        base[f"pred_small_s{s}"] = base[f"pred_nano_s{s}"] - base[f"pred_recovery_ns_s{s}"]
        base[f"pred_medium_s{s}"] = base[f"pred_small_s{s}"] - base[f"pred_recovery_sm_s{s}"]
    for tier in ["nano", "small", "medium"]:
        base[f"pred_{tier}_avg"] = base[[f"pred_{tier}_s{s}"
                                          for s in range(N_STEPS)]].mean(axis=1)
    for tgt in ["nano", "recovery_ns", "recovery_sm"]:
        base[f"true_{tgt}"] = base[[f"true_{tgt}_s{s}"
                                      for s in range(N_STEPS)]].mean(axis=1)
    base["true_small"] = base["true_nano"] - base["true_recovery_ns"]
    base["true_medium"] = base["true_small"] - base["true_recovery_sm"]
    print(f"  Total windows: {len(base)}", flush=True)
    return base


def train_test_split(df):
    tr, te = [], []
    for intx in df["intersection"].unique():
        sub = df[df["intersection"] == intx].sort_values("frame_idx")
        k = int(len(sub) * 0.60)
        tr.append(sub.iloc[:k]); te.append(sub.iloc[k:])
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)


def oracle_choice_arr(df, T):
    tn = df["true_nano"].values
    ts = df["true_small"].values
    tm = df["true_medium"].values
    oc = np.full(len(df), -1, dtype=int)
    oc[tm < T] = 2
    oc[ts < T] = 1
    oc[tn < T] = 0
    return oc


# ── Metrics ─────────────────────────────────────────────────────────

def compute_all_metrics(df, sels, T):
    """Return dict of all metrics + aux stats; all over SOLVABLE windows."""
    sels = np.asarray(sels, dtype=int)
    oc = oracle_choice_arr(df, T)
    solvable = oc >= 0
    n_tot = len(df); n_sol = int(solvable.sum())
    if n_sol == 0:
        return {"n_solvable": 0, "n_total": n_tot,
                "energy_savings_pct": 0, "adequate_rate": 0, "correct_rate": 0,
                "over_provision_rate": 0, "mean_miss_rate": 0,
                "ewq": 0.0, "mras": 0.0, "pareto": 0.0, "trmr": 0.0,
                "pct_nano": 0, "pct_small": 0, "pct_medium": 0,
                "switches_per_100": 0}

    ss = sels[solvable]; so = oc[solvable]
    sol_df = df[solvable].reset_index(drop=True)

    tn = sol_df["true_nano"].values
    tsm = sol_df["true_small"].values
    tmd = sol_df["true_medium"].values
    true_mr = np.where(ss == 0, tn, np.where(ss == 1, tsm, tmd))
    e = ENERGY[ss]
    e_ratio = e / E_MEDIUM
    energy_saved = 1.0 - e_ratio  # relative to medium

    # Binary adequacy / correctness / over-provision
    adequate = (ss >= so).astype(float)
    correct = (ss == so).astype(float)
    over_prov = (ss > so).astype(float)

    # EWQ
    quality = np.clip(1.0 - true_mr / T, 0.0, None)
    efficiency = 1.0 - e_ratio
    ewq = (quality * (1.0 + efficiency)).mean()

    # MRAS
    acceptable = true_mr < T
    mras = np.where(acceptable, energy_saved, -MRAS_PENALTY).mean()

    # Pareto
    adq_frac = adequate.mean()
    sav_frac = energy_saved.mean()  # mean((E_MED - E[s])/E_MED) over solvable
    pareto = adq_frac * sav_frac

    # TRMR
    headroom = T - true_mr
    trmr_vals = np.where(
        headroom >= 0,
        (1.0 - e_ratio) * (1.0 - headroom / T),
        headroom,
    )
    trmr = trmr_vals.mean()

    switches = int(np.sum(sels[1:] != sels[:-1])) if n_tot > 1 else 0

    return {
        "n_solvable": n_sol, "n_total": n_tot,
        "energy_savings_pct": round(float(sav_frac * 100), 4),
        "adequate_rate": round(float(adq_frac * 100), 4),
        "correct_rate": round(float(correct.mean() * 100), 4),
        "over_provision_rate": round(float(over_prov.mean() * 100), 4),
        "mean_miss_rate": round(float(true_mr.mean()), 6),
        "ewq": round(float(ewq), 6),
        "mras": round(float(mras), 6),
        "pareto": round(float(pareto), 6),
        "trmr": round(float(trmr), 6),
        "pct_nano": round(float((ss == 0).mean() * 100), 2),
        "pct_small": round(float((ss == 1).mean() * 100), 2),
        "pct_medium": round(float((ss == 2).mean() * 100), 2),
        "switches_per_100": round(switches / n_tot * 100, 2),
    }


def metric_score(m, metric):
    if metric == "binary":
        return m["adequate_rate"] / 100.0 + 0.5 * m["energy_savings_pct"] / 100.0
    if metric == "ewq":
        return m["ewq"]
    if metric == "mras":
        return m["mras"]
    if metric == "pareto":
        return m["pareto"]
    if metric == "trmr":
        return m["trmr"]
    raise ValueError(metric)


# ── Controllers ─────────────────────────────────────────────────────

def always_sels(df, m): return np.full(len(df), m, dtype=int)


def oracle_sels(df, T):
    oc = oracle_choice_arr(df, T)
    return np.where(oc >= 0, oc, 2)


def direct_threshold(df, T, mn, ms):
    na = df["pred_nano_avg"].values
    sa = df["pred_small_avg"].values
    sels = np.full(len(df), 2, dtype=int)
    sels[sa < T - ms] = 1
    sels[na < T - mn] = 0
    return sels


def energy_penalty(df, lu, T):
    e_norm = ENERGY / E_MEDIUM
    pn = df["pred_nano_avg"].values
    ps = df["pred_small_avg"].values
    pm = df["pred_medium_avg"].values
    costs = np.stack([
        e_norm[0] + lu * np.clip(pn - T, 0, None),
        e_norm[1] + lu * np.clip(ps - T, 0, None),
        e_norm[2] + lu * np.clip(pm - T, 0, None),
    ], axis=1)
    return costs.argmin(axis=1)


# ── Optimization ────────────────────────────────────────────────────

def optimize_dt(tr, T, metric):
    best_score = -1e18; best = (0.10, 0.10)
    for mn in MARGINS:
        for ms in MARGINS:
            m = compute_all_metrics(tr, direct_threshold(tr, T, mn, ms), T)
            sc = metric_score(m, metric)
            if sc > best_score:
                best_score = sc; best = (mn, ms)
    return best


def optimize_ep(tr, T, metric):
    best_score = -1e18; best_lu = 5.0
    for lu in LU_GRID:
        m = compute_all_metrics(tr, energy_penalty(tr, lu, T), T)
        sc = metric_score(m, metric)
        if sc > best_score:
            best_score = sc; best_lu = lu
    return best_lu


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("  ALTERNATIVE EVALUATION METRICS")
    print("=" * 72)
    df = load_predictions()
    all_data = pd.concat([df[df["split"] == "within"], df[df["split"] == "cross"]],
                          ignore_index=True)
    tr, te = train_test_split(all_data)
    print(f"  Train={len(tr)} Test={len(te)}")

    rows = []
    baseline_ctrls = [("AlwaysNano", lambda d, T: always_sels(d, 0)),
                       ("AlwaysSmall", lambda d, T: always_sels(d, 1)),
                       ("AlwaysMedium", lambda d, T: always_sels(d, 2)),
                       ("Oracle", lambda d, T: oracle_sels(d, T))]

    for T in THRESHOLDS:
        print(f"\n{'─'*72}\n  THRESHOLD = {T}\n{'─'*72}", flush=True)

        # Baselines: no tuning, report once with opt_metric="none".
        for name, fn in baseline_ctrls:
            sels = fn(te, T)
            m = compute_all_metrics(te, sels, T)
            m["threshold"] = T; m["controller"] = name; m["opt_metric"] = "none"
            m["config"] = ""
            rows.append(m)
            print(f"    {name:18s} [none]   : sav={m['energy_savings_pct']:5.1f}% "
                  f"adq={m['adequate_rate']:5.1f}% ewq={m['ewq']:.3f} "
                  f"mras={m['mras']:+.3f} par={m['pareto']:.3f} trmr={m['trmr']:+.3f}",
                  flush=True)

        # DirectThreshold × each metric.
        for metric in METRICS:
            mn, ms = optimize_dt(tr, T, metric)
            sels = direct_threshold(te, T, mn, ms)
            m = compute_all_metrics(te, sels, T)
            m["threshold"] = T; m["controller"] = "DirectThreshold"
            m["opt_metric"] = metric
            m["config"] = f"mn={mn:.2f},ms={ms:.2f}"
            rows.append(m)
            print(f"    DirectThreshold   [{metric:6s}] : "
                  f"sav={m['energy_savings_pct']:5.1f}% "
                  f"adq={m['adequate_rate']:5.1f}% ewq={m['ewq']:.3f} "
                  f"mras={m['mras']:+.3f} par={m['pareto']:.3f} "
                  f"trmr={m['trmr']:+.3f}  {m['config']}", flush=True)

        # EnergyPenalty × each metric.
        for metric in METRICS:
            lu = optimize_ep(tr, T, metric)
            sels = energy_penalty(te, lu, T)
            m = compute_all_metrics(te, sels, T)
            m["threshold"] = T; m["controller"] = "EnergyPenalty"
            m["opt_metric"] = metric
            m["config"] = f"lu={lu}"
            rows.append(m)
            print(f"    EnergyPenalty     [{metric:6s}] : "
                  f"sav={m['energy_savings_pct']:5.1f}% "
                  f"adq={m['adequate_rate']:5.1f}% ewq={m['ewq']:.3f} "
                  f"mras={m['mras']:+.3f} par={m['pareto']:.3f} "
                  f"trmr={m['trmr']:+.3f}  {m['config']}", flush=True)

    cols = ["threshold", "controller", "opt_metric", "config",
            "energy_savings_pct", "adequate_rate", "ewq", "mras",
            "pareto", "trmr", "correct_rate", "over_provision_rate",
            "mean_miss_rate", "pct_nano", "pct_small", "pct_medium",
            "switches_per_100", "n_solvable", "n_total"]
    out_df = pd.DataFrame(rows)[cols]
    out_tsv = SCRIPT_DIR / "controller_alt_metrics_results.tsv"
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n  Saved {out_tsv}", flush=True)

    # ── Summary table ──
    print("\n" + "=" * 76)
    print("  WINNERS BY METRIC (test split)")
    print("=" * 76)
    for T in THRESHOLDS:
        print(f"\n  T={T}:")
        print(f"    {'Metric':18s} | {'Winner':22s} | {'Score':>7s} | "
              f"{'DT best':>8s} | {'EP best':>8s}")
        print("    " + "-" * 74)
        tr_rows = [r for r in rows if r["threshold"] == T]
        for metric in METRICS:
            # For each controller, pick its row matched to this opt_metric,
            # then score it on this metric's value.
            candidates = []
            for r in tr_rows:
                if r["opt_metric"] == "none" or r["opt_metric"] == metric:
                    candidates.append(r)
            if not candidates:
                continue
            scored = [(metric_score(r, metric), r) for r in candidates]
            scored.sort(key=lambda x: -x[0])
            winner_sc, winner = scored[0]
            dt_best = max((metric_score(r, metric) for r in tr_rows
                            if r["controller"] == "DirectThreshold"
                            and r["opt_metric"] == metric), default=float("nan"))
            ep_best = max((metric_score(r, metric) for r in tr_rows
                            if r["controller"] == "EnergyPenalty"
                            and r["opt_metric"] == metric), default=float("nan"))
            print(f"    {metric:18s} | {winner['controller']:22s} | "
                  f"{winner_sc:7.3f} | {dt_best:8.3f} | {ep_best:8.3f}")

    # ── Plot: 2x2 subplots, one per threshold ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False)
    metric_colors = {"binary": "tab:blue", "ewq": "tab:orange",
                      "mras": "tab:green", "pareto": "tab:red",
                      "trmr": "tab:purple"}
    baseline_style = {"AlwaysNano": ("green", "D"), "AlwaysSmall": ("orange", "D"),
                       "AlwaysMedium": ("purple", "D"), "Oracle": ("black", "*")}

    for ax, T in zip(axes.flat, THRESHOLDS):
        tr_rows = [r for r in rows if r["threshold"] == T]
        # Baselines
        for name, (c, mk) in baseline_style.items():
            r = next((r for r in tr_rows if r["controller"] == name), None)
            if r is None:
                continue
            ax.scatter(r["adequate_rate"], r["energy_savings_pct"],
                        c=c, marker=mk, s=160 if mk == "*" else 90,
                        edgecolor="black", linewidth=0.6, zorder=6)
            ax.annotate(name, (r["adequate_rate"], r["energy_savings_pct"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
        # DT / EP per metric
        for r in tr_rows:
            if r["opt_metric"] == "none":
                continue
            c = metric_colors.get(r["opt_metric"], "gray")
            marker = "P" if r["controller"] == "DirectThreshold" else "h"
            ax.scatter(r["adequate_rate"], r["energy_savings_pct"],
                        c=c, marker=marker, s=110,
                        edgecolor="black", linewidth=0.4, zorder=5,
                        label=f"{r['controller'][:2]}[{r['opt_metric']}]")
        ax.set_title(f"T={T}  ({tr_rows[0]['n_solvable']}/{tr_rows[0]['n_total']} solvable)")
        ax.set_xlabel("Adequate rate (%)")
        ax.set_ylabel("Energy savings (%)")
        ax.grid(True, alpha=0.3)

    # Dedupe legend entries
    handles_list, labels_list = axes[0, 0].get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles_list, labels_list):
        if lab not in seen:
            seen[lab] = h
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, fontsize=8,
                bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plot_path = SCRIPT_DIR / "metric_comparison_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {plot_path}", flush=True)

    print(f"\n  Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
