#!/usr/bin/env python3
"""Temporal-aware single-decision controllers driven by per-step predictions.

Controllers (all emit one selection per window):
  1. DirectThreshold       — averaged preds (baseline)
  2. WorstCaseThreshold    — max across 5 steps
  3. WeightedThreshold     — exp-decay weighted mean across 5 steps
  4. VarianceAwareThreshold — mean + gamma * std across 5 steps
  5. AnyStepThreshold      — "all 5 steps adequate"
  6. EnergyPenalty         — greedy Bayes on avg preds (stateless)
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

E_NANO = 15 * 85.36
E_SMALL = 15 * 128.65
E_MEDIUM = 15 * 248.46
ENERGY = {0: E_NANO, 1: E_SMALL, 2: E_MEDIUM}
THRESHOLDS = [0.30, 0.35, 0.40, 0.50]
N_STEPS = 5
SEEDS = [42, 43, 44]
WEIGHTS = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
assert abs(WEIGHTS.sum() - 1.0) < 1e-9


# ── Data ────────────────────────────────────────────────────────────

def load_predictions():
    dfs = [pd.read_csv(SCRIPT_DIR / f"predictions_multistep_seed{s}.csv")
           for s in SEEDS]
    true_cols = [c for c in dfs[0].columns if c.startswith("true_")]
    for c in true_cols:
        v = [df[c].values for df in dfs]
        for i in range(1, len(v)):
            assert np.allclose(v[0], v[i], atol=1e-8), f"true differ: {c}"
    print("  True values identical across 3 seeds.", flush=True)

    base = dfs[0][["frame_idx", "split", "intersection"]].copy()
    for c in [c for c in dfs[0].columns if c.startswith("pred_")]:
        base[c] = np.mean([df[c].values for df in dfs], axis=0)
    for c in true_cols:
        base[c] = dfs[0][c].values

    # Per-tier per-step miss rate predictions.
    for s in range(N_STEPS):
        base[f"pred_small_s{s}"] = base[f"pred_nano_s{s}"] - base[f"pred_recovery_ns_s{s}"]
        base[f"pred_medium_s{s}"] = base[f"pred_small_s{s}"] - base[f"pred_recovery_sm_s{s}"]

    # Averaged per-tier miss rates (for DirectThreshold, EnergyPenalty).
    for tier in ["nano", "small", "medium"]:
        base[f"pred_{tier}_avg"] = base[[f"pred_{tier}_s{s}"
                                          for s in range(N_STEPS)]].mean(axis=1)

    # True per-tier miss rates used by oracle/evaluation.
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


# ── Helper: per-step stacks for fast access ────────────────────────

def stack_steps(df, tier):
    """(N, 5) array for `pred_{tier}_s0..s4`."""
    return df[[f"pred_{tier}_s{s}" for s in range(N_STEPS)]].values


# ── Baselines ───────────────────────────────────────────────────────

def always(df, m): return [m] * len(df)

def oracle_ctrl(df, T):
    _, oc = get_solvable(df, T)
    return [int(c) if c >= 0 else 2 for c in oc]


# ── Controller 1: DirectThreshold ───────────────────────────────────

def direct_threshold(df, T, mn, ms):
    na = df["pred_nano_avg"].values
    sa = df["pred_small_avg"].values
    sels = np.full(len(df), 2, dtype=int)
    sels[sa < T - ms] = 1
    sels[na < T - mn] = 0
    return sels.tolist()


# ── Controller 2: WorstCaseThreshold (max across steps) ────────────

def worst_case_threshold(df, T, mn, ms):
    nw = stack_steps(df, "nano").max(axis=1)
    sw = stack_steps(df, "small").max(axis=1)
    sels = np.full(len(df), 2, dtype=int)
    sels[sw < T - ms] = 1
    sels[nw < T - mn] = 0
    return sels.tolist()


# ── Controller 3: WeightedThreshold (exp-decay weights) ────────────

def weighted_threshold(df, T, mn, ms):
    nw = stack_steps(df, "nano") @ WEIGHTS
    sw = stack_steps(df, "small") @ WEIGHTS
    sels = np.full(len(df), 2, dtype=int)
    sels[sw < T - ms] = 1
    sels[nw < T - mn] = 0
    return sels.tolist()


# ── Controller 4: VarianceAwareThreshold ───────────────────────────

def variance_aware_threshold(df, T, mn, ms, gamma):
    ns = stack_steps(df, "nano")
    ss = stack_steps(df, "small")
    n_adj = ns.mean(axis=1) + gamma * ns.std(axis=1, ddof=0)
    s_adj = ss.mean(axis=1) + gamma * ss.std(axis=1, ddof=0)
    sels = np.full(len(df), 2, dtype=int)
    sels[s_adj < T - ms] = 1
    sels[n_adj < T - mn] = 0
    return sels.tolist()


# ── Controller 5: AnyStepThreshold ─────────────────────────────────

def any_step_threshold(df, T, mn, ms):
    ns = stack_steps(df, "nano")
    ss = stack_steps(df, "small")
    all_nano_ok = (ns < T - mn).all(axis=1)
    all_small_ok = (ss < T - ms).all(axis=1)
    sels = np.full(len(df), 2, dtype=int)
    sels[all_small_ok] = 1
    sels[all_nano_ok] = 0
    return sels.tolist()


# ── Controller 6: EnergyPenalty (stateless greedy on avg) ──────────

def energy_penalty(df, lu, T):
    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}
    pn = df["pred_nano_avg"].values
    ps = df["pred_small_avg"].values
    pm = df["pred_medium_avg"].values
    sels = []
    for i in range(len(df)):
        pms = {0: pn[i], 1: ps[i], 2: pm[i]}
        best_c = float("inf"); best_m = 2
        for m in range(3):
            c = e_norm[m] + lu * max(0.0, pms[m] - T)
            if c < best_c:
                best_c = c; best_m = m
        sels.append(best_m)
    return sels


# ── Optimization ────────────────────────────────────────────────────

MARGINS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
LU_GRID = [1.0, 3.0, 5.0, 10.0, 20.0]
GAMMA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]


def _score(m):
    return m["adequate_rate"]/100 + 0.5 * m["energy_savings_pct"]/100


def optimize_margin_ctrl(fn, tr, T, extra_grid=None):
    """Grid search over (mn, ms) plus optional extra_grid of (key, values)."""
    best_score = -1; best = None
    if extra_grid is None:
        for mn in MARGINS:
            for ms in MARGINS:
                m = evaluate(tr, fn(tr, T, mn, ms), T)
                s = _score(m)
                if s > best_score:
                    best_score = s; best = {"mn": mn, "ms": ms}
        return best
    key, values = extra_grid
    for mn in MARGINS:
        for ms in MARGINS:
            for v in values:
                m = evaluate(tr, fn(tr, T, mn, ms, v), T)
                s = _score(m)
                if s > best_score:
                    best_score = s; best = {"mn": mn, "ms": ms, key: v}
    return best


def optimize_energy_penalty(tr, T):
    best_score = -1; best_lu = 5.0
    for lu in LU_GRID:
        m = evaluate(tr, energy_penalty(tr, lu, T), T)
        s = _score(m)
        if s > best_score:
            best_score = s; best_lu = lu
    return best_lu


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 72)
    print("  TEMPORAL-AWARE SINGLE-DECISION CONTROLLERS")
    print("=" * 72)
    df = load_predictions()
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
            print(f"    {name:52s} sav={m['energy_savings_pct']:5.1f}% "
                  f"adq={m['adequate_rate']:5.1f}% "
                  f"sw/100={m['switches_per_100']:5.2f}", flush=True)
            return m

        record("AlwaysNano", always(te, 0))
        record("AlwaysSmall", always(te, 1))
        record("AlwaysMedium", always(te, 2))
        record("Oracle", oracle_ctrl(te, T))

        print("    Optimizing DirectThreshold...", flush=True)
        cfg = optimize_margin_ctrl(direct_threshold, tr, T)
        record(f"DirectThreshold(mn={cfg['mn']:.2f},ms={cfg['ms']:.2f})",
               direct_threshold(te, T, cfg["mn"], cfg["ms"]))

        print("    Optimizing WorstCaseThreshold...", flush=True)
        cfg = optimize_margin_ctrl(worst_case_threshold, tr, T)
        record(f"WorstCaseThreshold(mn={cfg['mn']:.2f},ms={cfg['ms']:.2f})",
               worst_case_threshold(te, T, cfg["mn"], cfg["ms"]))

        print("    Optimizing WeightedThreshold...", flush=True)
        cfg = optimize_margin_ctrl(weighted_threshold, tr, T)
        record(f"WeightedThreshold(mn={cfg['mn']:.2f},ms={cfg['ms']:.2f})",
               weighted_threshold(te, T, cfg["mn"], cfg["ms"]))

        print("    Optimizing VarianceAwareThreshold...", flush=True)
        cfg = optimize_margin_ctrl(variance_aware_threshold, tr, T,
                                    extra_grid=("gamma", GAMMA_GRID))
        record(f"VarianceAwareThreshold(mn={cfg['mn']:.2f},"
               f"ms={cfg['ms']:.2f},g={cfg['gamma']})",
               variance_aware_threshold(te, T, cfg["mn"], cfg["ms"], cfg["gamma"]))

        print("    Optimizing AnyStepThreshold...", flush=True)
        cfg = optimize_margin_ctrl(any_step_threshold, tr, T)
        record(f"AnyStepThreshold(mn={cfg['mn']:.2f},ms={cfg['ms']:.2f})",
               any_step_threshold(te, T, cfg["mn"], cfg["ms"]))

        print("    Optimizing EnergyPenalty...", flush=True)
        lu = optimize_energy_penalty(tr, T)
        record(f"EnergyPenalty(lu={lu})", energy_penalty(te, lu, T))

    cols = ["threshold", "n_solvable", "n_total", "controller",
            "energy_savings_pct", "adequate_rate", "correct_rate",
            "over_provision_rate", "mean_miss_rate",
            "pct_nano", "pct_small", "pct_medium", "switches_per_100"]
    out_df = pd.DataFrame(rows)[cols]
    out_tsv = SCRIPT_DIR / "controller_temporal_results.tsv"
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n  Saved {out_tsv}", flush=True)

    # ── Comparison plot (T=0.50) ──
    T_plot = 0.50
    tr_rows = [r for r in rows if r["threshold"] == T_plot]
    styles = {
        "AlwaysNano": ("green", "D", 100),
        "AlwaysSmall": ("orange", "D", 100),
        "AlwaysMedium": ("purple", "D", 100),
        "Oracle": ("black", "*", 200),
    }
    prefix_styles = {
        "DirectThreshold": ("limegreen", "P", 140, "DirectThreshold"),
        "WorstCaseThreshold": ("red", "v", 140, "WorstCaseThreshold"),
        "WeightedThreshold": ("blue", "s", 140, "WeightedThreshold"),
        "VarianceAwareThreshold": ("magenta", "^", 150, "VarianceAwareThreshold"),
        "AnyStepThreshold": ("brown", "X", 140, "AnyStepThreshold"),
        "EnergyPenalty": ("teal", "h", 130, "EnergyPenalty"),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
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
                    ax.annotate(f"sw={r['switches_per_100']:.1f}",
                                 (r["energy_savings_pct"], r["adequate_rate"]),
                                 xytext=(5, 5), textcoords="offset points",
                                 fontsize=7)
                    break
    ax.set_xlabel("Energy Savings (%)")
    ax.set_ylabel("Adequate Rate (%, solvable only)")
    n_s = tr_rows[0]["n_solvable"]; n_t = tr_rows[0]["n_total"]
    ax.set_title(f"Temporal-aware controllers at T={T_plot} "
                  f"({n_s}/{n_t} solvable) — labels: switches/100")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plot_path = SCRIPT_DIR / "temporal_comparison_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved {plot_path}", flush=True)

    # ── Summary table ──
    print("\n" + "=" * 110)
    print("  Controller comparison (savings% / adequate% / switches_per_100)")
    print("=" * 110)
    key_ctrls = [("Oracle", "Oracle"),
                  ("DirectThresh", "DirectThreshold"),
                  ("WorstCase", "WorstCaseThreshold"),
                  ("Weighted", "WeightedThreshold"),
                  ("VarAware", "VarianceAwareThreshold"),
                  ("AnyStep", "AnyStepThreshold"),
                  ("EnergyPenalty", "EnergyPenalty")]
    header = f"  {'T':>5s} | " + " | ".join(f"{lbl[:14]:>14s}" for lbl, _ in key_ctrls)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for T in THRESHOLDS:
        ts = [r for r in rows if r["threshold"] == T]
        cells = []
        for lbl, pre in key_ctrls:
            hit = next((r for r in ts
                         if r["controller"] == pre
                         or r["controller"].startswith(pre + "(")), None)
            if hit is None:
                cells.append(f"{'—':>14s}")
            else:
                cells.append(f"{hit['energy_savings_pct']:4.1f}/"
                             f"{hit['adequate_rate']:4.1f}/"
                             f"{hit['switches_per_100']:4.1f}".rjust(14))
        print(f"  {T:>5.2f} | " + " | ".join(cells))

    print(f"\n  Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
