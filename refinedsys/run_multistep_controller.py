#!/usr/bin/env python3
"""Part 2: Multi-step MPC controller using per-step LSTM predictions.

Consumes refinedsys/predictions_multistep_seed{42,43,44}.csv (3-seed avg).

Controllers:
  - AlwaysNano / AlwaysSmall / AlwaysMedium / Oracle (baselines)
  - DirectThreshold (stateless, uses s0 or pred-avg — uses avg here)
  - GreedyBayes (stateless, H=1; existing BayesMPC with H=1, w_s=0)
  - MultiStepMPC (H=5, per-step preds, switching cost, state tracking)
  - MultiStepMPC_RichVariant (adds over-provision penalty lo)
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
PROJECT_DIR = SCRIPT_DIR.parent

E_NANO = 15 * 85.36
E_SMALL = 15 * 128.65
E_MEDIUM = 15 * 248.46
ENERGY = {0: E_NANO, 1: E_SMALL, 2: E_MEDIUM}
THRESHOLDS = [0.30, 0.35, 0.40, 0.50]
N_STEPS = 5
SEEDS = [42, 43, 44]


# ── Data ────────────────────────────────────────────────────────────

def load_multistep_predictions():
    """Average per-step predictions across 3 seeds; verify true identical."""
    dfs = [pd.read_csv(SCRIPT_DIR / f"predictions_multistep_seed{s}.csv")
           for s in SEEDS]
    for col in [f"true_{t}_s{s}" for t in ("nano", "recovery_ns", "recovery_sm")
                for s in range(N_STEPS)]:
        v = [df[col].values for df in dfs]
        for i in range(1, len(v)):
            assert np.allclose(v[0], v[i], atol=1e-8), f"true differ: {col}"
    print("  True values identical across 3 seeds.", flush=True)

    base = dfs[0][["frame_idx", "split", "intersection"]].copy()
    # Average pred columns
    pred_cols = [c for c in dfs[0].columns if c.startswith("pred_")]
    for c in pred_cols:
        base[c] = np.mean([df[c].values for df in dfs], axis=0)
    # Copy true columns
    true_cols = [c for c in dfs[0].columns if c.startswith("true_")]
    for c in true_cols:
        base[c] = dfs[0][c].values

    # Build per-tier miss-rate predictions per step.
    for s in range(N_STEPS):
        base[f"pred_small_s{s}"] = base[f"pred_nano_s{s}"] - base[f"pred_recovery_ns_s{s}"]
        base[f"pred_medium_s{s}"] = base[f"pred_small_s{s}"] - base[f"pred_recovery_sm_s{s}"]
    # Averaged (backward-compat) per-tier miss rates for stateless controllers.
    base["pred_nano_avg"] = base[[f"pred_nano_s{s}" for s in range(N_STEPS)]].mean(axis=1)
    base["pred_small_avg"] = (base["pred_nano_avg"]
                               - base[[f"pred_recovery_ns_s{s}"
                                        for s in range(N_STEPS)]].mean(axis=1))
    base["pred_medium_avg"] = (base["pred_small_avg"]
                                - base[[f"pred_recovery_sm_s{s}"
                                         for s in range(N_STEPS)]].mean(axis=1))

    # True per-tier miss rates (averaged across steps — matches evaluation target).
    base["true_nano"] = base[[f"true_nano_s{s}" for s in range(N_STEPS)]].mean(axis=1)
    base["true_recovery_ns"] = base[[f"true_recovery_ns_s{s}"
                                       for s in range(N_STEPS)]].mean(axis=1)
    base["true_recovery_sm"] = base[[f"true_recovery_sm_s{s}"
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


def oracle_choice(row, T):
    if row["true_nano"] < T: return 0
    if row["true_small"] < T: return 1
    if row["true_medium"] < T: return 2
    return -1


def get_solvable(df, T):
    oc = df.apply(lambda r: oracle_choice(r, T), axis=1).values
    return oc >= 0, oc


def evaluate(df, sels, T):
    solvable, oc = get_solvable(df, T)
    n_tot = len(df); n_sol = int(solvable.sum())
    if n_sol == 0:
        return {"n_solvable": 0, "n_total": n_tot, "energy_savings_pct": 0,
                "adequate_rate": 0, "correct_rate": 0, "over_provision_rate": 0,
                "mean_miss_rate": 0, "pct_nano": 0, "pct_small": 0,
                "pct_medium": 0, "switches_per_100": 0}
    arr = np.array(sels)
    ss = arr[solvable]; so = oc[solvable]
    energies = np.array([ENERGY[s] for s in ss])
    sol_df = df[solvable].reset_index(drop=True)
    tm = np.array([sol_df.iloc[i]["true_nano"] if ss[i] == 0
                    else sol_df.iloc[i]["true_small"] if ss[i] == 1
                    else sol_df.iloc[i]["true_medium"]
                    for i in range(n_sol)])
    switches = sum(1 for i in range(1, n_tot) if sels[i] != sels[i-1])
    return {
        "n_solvable": n_sol, "n_total": n_tot,
        "energy_savings_pct": round((1 - energies.mean() / E_MEDIUM) * 100, 2),
        "adequate_rate": round((ss >= so).mean() * 100, 2),
        "correct_rate": round((ss == so).mean() * 100, 2),
        "over_provision_rate": round((ss > so).mean() * 100, 2),
        "mean_miss_rate": round(float(tm.mean()), 6),
        "pct_nano": round((ss == 0).mean() * 100, 1),
        "pct_small": round((ss == 1).mean() * 100, 1),
        "pct_medium": round((ss == 2).mean() * 100, 1),
        "switches_per_100": round(switches / n_tot * 100, 2),
    }


# ── Baselines ───────────────────────────────────────────────────────

def always(df, m): return [m] * len(df)

def oracle_ctrl(df, T):
    _, oc = get_solvable(df, T)
    return [int(c) if c >= 0 else 2 for c in oc]


# ── Stateless: DirectThreshold (uses avg preds) ─────────────────────

def direct_threshold(df, T, mn, ms):
    sels = []
    for _, r in df.iterrows():
        if r["pred_nano_avg"] < T - mn: sels.append(0)
        elif r["pred_small_avg"] < T - ms: sels.append(1)
        else: sels.append(2)
    return sels


def optimize_direct(tr, T):
    margins = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    best_score = -1; best = (0.10, 0.10)
    for mn in margins:
        for ms in margins:
            m = evaluate(tr, direct_threshold(tr, T, mn, ms), T)
            score = m["adequate_rate"]/100 + 0.5 * m["energy_savings_pct"]/100
            if score > best_score:
                best_score = score; best = (mn, ms)
    return best


# ── Stateless: GreedyBayes (H=1, no switch, no state) ───────────────

def greedy_bayes(df, lu, T):
    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}
    sels = []
    for _, row in df.iterrows():
        pm = {0: row["pred_nano_avg"], 1: row["pred_small_avg"],
              2: row["pred_medium_avg"]}
        best_cost = float("inf"); best_m = 2
        for m in range(3):
            c = e_norm[m] + lu * max(0, pm[m] - T)
            if c < best_cost:
                best_cost = c; best_m = m
        sels.append(best_m)
    return sels


def optimize_greedy(tr, T):
    best_adq = -1; best_lu = 5.0
    for lu in [1.0, 3.0, 5.0, 10.0]:
        m = evaluate(tr, greedy_bayes(tr, lu, T), T)
        if m["energy_savings_pct"] > 10 and m["adequate_rate"] > best_adq:
            best_adq = m["adequate_rate"]; best_lu = lu
    if best_adq < 0:
        for lu in [1.0, 3.0, 5.0, 10.0]:
            m = evaluate(tr, greedy_bayes(tr, lu, T), T)
            if m["adequate_rate"] > best_adq:
                best_adq = m["adequate_rate"]; best_lu = lu
    return best_lu


# ── Multi-step MPC (stateful, per-step preds, switching cost) ───────

def _build_per_step_arrays(df):
    """Precompute (N, 5, 3) per-tier miss-rate prediction arrays."""
    n = len(df)
    arr = np.zeros((n, N_STEPS, 3), dtype=np.float64)
    for s in range(N_STEPS):
        arr[:, s, 0] = df[f"pred_nano_s{s}"].values
        arr[:, s, 1] = df[f"pred_small_s{s}"].values
        arr[:, s, 2] = df[f"pred_medium_s{s}"].values
    return arr


def multistep_mpc(df, lu, w_switch, T, lo=0.0, init_model=1):
    """H=5, per-step preds, switching cost. If lo>0, adds over-provision term."""
    seqs = list(itertools.product(range(3), repeat=N_STEPS))
    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}
    e_rel_nano = {m: ENERGY[m] / E_NANO for m in range(3)}
    per_step = _build_per_step_arrays(df)

    sels = []
    current = init_model
    n = len(df)
    for i in range(n):
        best_cost = float("inf"); best_first = current
        pm = per_step[i]  # (5, 3)
        for seq in seqs:
            cost = 0.0
            prev = current
            for step in range(N_STEPS):
                m = seq[step]
                cost += e_norm[m]
                cost += lu * max(0.0, pm[step, m] - T)
                if lo > 0.0:
                    cost += lo * max(0.0, T - pm[step, m]) * e_rel_nano[m]
                if m != prev:
                    cost += w_switch
                prev = m
            if cost < best_cost:
                best_cost = cost; best_first = seq[0]
        sels.append(best_first)
        current = best_first
    return sels


def optimize_multistep(tr, T, use_rich=False):
    lu_vals = [1.0, 3.0, 5.0, 10.0, 20.0]
    ws_vals = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    lo_vals = [0.1, 0.3, 0.5, 1.0] if use_rich else [0.0]
    best_score = -1; best = None
    for lu in lu_vals:
        for ws in ws_vals:
            for lo in lo_vals:
                m = evaluate(tr, multistep_mpc(tr, lu, ws, T, lo=lo), T)
                # Score: adequacy + 0.3*savings (same weighting as existing optimizers)
                score = m["adequate_rate"]/100 + 0.3 * m["energy_savings_pct"]/100
                if score > best_score:
                    best_score = score
                    best = {"lambda_u": lu, "w_switch": ws, "lambda_o": lo}
    return best


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("  MULTI-STEP MPC CONTROLLER (per-step preds, state, switching cost)")
    print("=" * 72)
    df = load_multistep_predictions()
    all_data = pd.concat([df[df["split"] == "within"], df[df["split"] == "cross"]],
                          ignore_index=True)
    tr, te = train_test_split(all_data)
    print(f"  Train={len(tr)} Test={len(te)}")

    rows = []
    for T in THRESHOLDS:
        print(f"\n{'─'*72}\n  THRESHOLD = {T}\n{'─'*72}", flush=True)
        sol, _ = get_solvable(te, T)
        print(f"  Solvable: {int(sol.sum())}/{len(te)}", flush=True)

        def record(name, sels):
            m = evaluate(te, sels, T)
            m["threshold"] = T; m["controller"] = name
            rows.append(m)
            print(f"    {name:44s} sav={m['energy_savings_pct']:5.1f}% "
                  f"adq={m['adequate_rate']:5.1f}% "
                  f"sw/100={m['switches_per_100']:5.2f}", flush=True)
            return m

        record("AlwaysNano", always(te, 0))
        record("AlwaysSmall", always(te, 1))
        record("AlwaysMedium", always(te, 2))
        record("Oracle", oracle_ctrl(te, T))

        print("    Optimizing DirectThreshold...", flush=True)
        mn, ms = optimize_direct(tr, T)
        record(f"DirectThreshold(mn={mn:.2f},ms={ms:.2f})",
               direct_threshold(te, T, mn, ms))

        print("    Optimizing GreedyBayes (H=1, stateless)...", flush=True)
        lu_g = optimize_greedy(tr, T)
        record(f"GreedyBayes(lu={lu_g})", greedy_bayes(te, lu_g, T))

        print("    Optimizing MultiStepMPC (H=5, stateful)...", flush=True)
        mcfg = optimize_multistep(tr, T, use_rich=False)
        record(f"MultiStepMPC(lu={mcfg['lambda_u']},ws={mcfg['w_switch']})",
               multistep_mpc(te, mcfg["lambda_u"], mcfg["w_switch"], T))

        print("    Optimizing MultiStepMPC_RichVariant...", flush=True)
        rcfg = optimize_multistep(tr, T, use_rich=True)
        record(f"MultiStepMPC_Rich(lu={rcfg['lambda_u']},ws={rcfg['w_switch']},"
               f"lo={rcfg['lambda_o']})",
               multistep_mpc(te, rcfg["lambda_u"], rcfg["w_switch"], T,
                              lo=rcfg["lambda_o"]))

    cols = ["threshold", "n_solvable", "n_total", "controller",
            "energy_savings_pct", "adequate_rate", "correct_rate",
            "over_provision_rate", "mean_miss_rate",
            "pct_nano", "pct_small", "pct_medium", "switches_per_100"]
    out_df = pd.DataFrame(rows)[cols]
    out_tsv = SCRIPT_DIR / "controller_multistep_results.tsv"
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n  Saved {out_tsv}", flush=True)

    # ── Comparison plot at T=0.50 ──
    T_plot = 0.50
    tr_rows = [r for r in rows if r["threshold"] == T_plot]
    styles = {
        "AlwaysNano": ("green", "D", 100),
        "AlwaysSmall": ("orange", "D", 100),
        "AlwaysMedium": ("purple", "D", 100),
        "Oracle": ("black", "*", 200),
    }
    prefix_styles = {
        "DirectThreshold": ("limegreen", "P", 140, "DirectThreshold (stateless)"),
        "GreedyBayes": ("red", "^", 120, "GreedyBayes (H=1 stateless)"),
        "MultiStepMPC_Rich": ("magenta", "v", 160, "MultiStepMPC+Rich (H=5 stateful)"),
        "MultiStepMPC": ("blue", "s", 140, "MultiStepMPC (H=5 stateful)"),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in tr_rows:
        name = r["controller"]
        if name in styles:
            c, mk, sz = styles[name]
            ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                        c=c, s=sz, marker=mk, zorder=6, label=name,
                        edgecolor="black", linewidth=0.5)
        else:
            for pre, (c, mk, sz, lab) in prefix_styles.items():
                if name.startswith(pre):
                    ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                                c=c, s=sz, marker=mk, zorder=6, label=lab,
                                edgecolor="black", linewidth=0.5)
                    # Annotate switches_per_100
                    ax.annotate(f"sw={r['switches_per_100']:.1f}",
                                 (r["energy_savings_pct"], r["adequate_rate"]),
                                 xytext=(5, 5), textcoords="offset points",
                                 fontsize=7)
                    break
    ax.set_xlabel("Energy Savings (%)")
    ax.set_ylabel("Adequate Rate (%, solvable only)")
    n_s = tr_rows[0]["n_solvable"]; n_t = tr_rows[0]["n_total"]
    ax.set_title(f"Controller comparison at T={T_plot} "
                  f"({n_s}/{n_t} solvable) — labels: switches/100")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plot_path = SCRIPT_DIR / "comparison_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved {plot_path}", flush=True)

    # ── Summary table ──
    print("\n" + "=" * 92)
    print("  Controller comparison (savings% / adequate% / switches_per_100)")
    print("=" * 92)
    key_ctrls = ["Oracle", "DirectThreshold", "GreedyBayes",
                 "MultiStepMPC(", "MultiStepMPC_Rich"]
    header = f"  {'T':>5s} | " + " | ".join(f"{k[:18]:>18s}" for k in key_ctrls)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for T in THRESHOLDS:
        ts = [r for r in rows if r["threshold"] == T]
        cells = []
        for k in key_ctrls:
            if k == "MultiStepMPC(":
                hit = next((r for r in ts
                             if r["controller"].startswith("MultiStepMPC(")), None)
            elif k == "MultiStepMPC_Rich":
                hit = next((r for r in ts
                             if r["controller"].startswith("MultiStepMPC_Rich")), None)
            else:
                hit = next((r for r in ts
                             if r["controller"].startswith(k) or r["controller"] == k),
                            None)
            if hit is None:
                cells.append(f"{'—':>18s}")
            else:
                cells.append(f"{hit['energy_savings_pct']:4.1f}/"
                             f"{hit['adequate_rate']:4.1f}/"
                             f"{hit['switches_per_100']:4.1f}".rjust(18))
        print(f"  {T:>5.2f} | " + " | ".join(cells))

    print(f"\n  Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
