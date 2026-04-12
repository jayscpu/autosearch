#!/usr/bin/env python3
"""Controller simulation driven by BEST_k80 predictor (3-seed avg)."""
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_TSV = SCRIPT_DIR.parent / "controller_rerun" / "results.tsv"

E_NANO = 15 * 85.36
E_SMALL = 15 * 128.65
E_MEDIUM = 15 * 248.46
ENERGY = {0: E_NANO, 1: E_SMALL, 2: E_MEDIUM}
THRESHOLDS = [0.30, 0.35, 0.40, 0.50]


def load_predictions():
    seeds = [42, 43, 44]
    dfs = [pd.read_csv(SCRIPT_DIR / f"predictions_best_seed{s}.csv") for s in seeds]
    for col in ["true_nano", "true_recovery_ns", "true_recovery_sm"]:
        vals = [df[col].values for df in dfs]
        for i in range(1, len(vals)):
            assert np.allclose(vals[0], vals[i], atol=1e-8), f"True differ: {col}"
    print("  True values identical across 3 seeds.", flush=True)
    base = dfs[0][["frame_idx", "split", "intersection",
                    "true_nano", "true_recovery_ns", "true_recovery_sm"]].copy()
    for c in ["pred_nano", "pred_recovery_ns", "pred_recovery_sm"]:
        base[c] = np.mean([df[c].values for df in dfs], axis=0)
    base["true_small"] = base["true_nano"] - base["true_recovery_ns"]
    base["true_medium"] = base["true_small"] - base["true_recovery_sm"]
    base["pred_small"] = base["pred_nano"] - base["pred_recovery_ns"]
    base["pred_medium"] = base["pred_small"] - base["pred_recovery_sm"]
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
    adequate = (ss >= so).mean()
    correct = (ss == so).mean()
    over = (ss > so).mean()
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
        "adequate_rate": round(adequate * 100, 2),
        "correct_rate": round(correct * 100, 2),
        "over_provision_rate": round(over * 100, 2),
        "mean_miss_rate": round(float(tm.mean()), 6),
        "pct_nano": round((ss == 0).mean() * 100, 1),
        "pct_small": round((ss == 1).mean() * 100, 1),
        "pct_medium": round((ss == 2).mean() * 100, 1),
        "switches_per_100": round(switches / n_tot * 100, 2),
    }


# ── Controllers ─────────────────────────────────────────────────────

def always(df, m): return [m] * len(df)

def oracle_ctrl(df, T):
    _, oc = get_solvable(df, T)
    return [int(c) if c >= 0 else 2 for c in oc]


def direct_threshold(df, T, mn, ms):
    sels = []
    for _, r in df.iterrows():
        if r["pred_nano"] < T - mn: sels.append(0)
        elif r["pred_small"] < T - ms: sels.append(1)
        else: sels.append(2)
    return sels


def optimize_direct(train_df, T):
    margins = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    best_score = -1; best = (0.10, 0.10)
    for mn in margins:
        for ms in margins:
            m = evaluate(train_df, direct_threshold(train_df, T, mn, ms), T)
            score = m["adequate_rate"]/100 + 0.5 * m["energy_savings_pct"]/100
            if score > best_score:
                best_score = score; best = (mn, ms)
    return best


def bayes_mpc(df, H, lu, T):
    seqs = list(itertools.product(range(3), repeat=H))
    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}
    sels = []
    rows = df.to_dict("records")
    for row in rows:
        pm = {0: row["pred_nano"], 1: row["pred_small"], 2: row["pred_medium"]}
        best_cost = float("inf"); best_first = 2
        for seq in seqs:
            cost = 0.0
            for step in range(H):
                m = seq[step]
                cost += e_norm[m]
                cost += lu * max(0, pm[m] - T)
            if cost < best_cost:
                best_cost = cost; best_first = seq[0]
        sels.append(best_first)
    return sels


def optimize_bayes(train_df, T):
    best_adq = -1; best = {"H": 1, "lambda_u": 5.0}
    for H in [1, 2, 3]:
        for lu in [1.0, 3.0, 5.0, 10.0]:
            m = evaluate(train_df, bayes_mpc(train_df, H, lu, T), T)
            if m["energy_savings_pct"] > 10 and m["adequate_rate"] > best_adq:
                best_adq = m["adequate_rate"]; best = {"H": H, "lambda_u": lu}
    if best_adq < 0:
        for H in [1, 2, 3]:
            for lu in [1.0, 3.0, 5.0, 10.0]:
                m = evaluate(train_df, bayes_mpc(train_df, H, lu, T), T)
                if m["adequate_rate"] > best_adq:
                    best_adq = m["adequate_rate"]; best = {"H": H, "lambda_u": lu}
    return best


def rich_mpc(df, lu, lo, T):
    e_norm = {m: ENERGY[m] / E_MEDIUM for m in range(3)}
    e_rel_nano = {m: ENERGY[m] / E_NANO for m in range(3)}
    sels = []
    for _, row in df.iterrows():
        pm = {0: row["pred_nano"], 1: row["pred_small"], 2: row["pred_medium"]}
        best_cost = float("inf"); best_m = 2
        for m in range(3):
            cost = e_norm[m]
            cost += lu * max(0, pm[m] - T)
            cost += lo * max(0, T - pm[m]) * e_rel_nano[m]
            if cost < best_cost:
                best_cost = cost; best_m = m
        sels.append(best_m)
    return sels


def optimize_rich(train_df, T):
    best_score = -1; best = {"lambda_u": 5.0, "lambda_o": 0.5}
    for lu in [1.0, 3.0, 5.0, 10.0, 20.0]:
        for lo in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            m = evaluate(train_df, rich_mpc(train_df, lu, lo, T), T)
            score = m["adequate_rate"]/100 + 0.3 * m["energy_savings_pct"]/100
            if score > best_score:
                best_score = score; best = {"lambda_u": lu, "lambda_o": lo}
    return best


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  CONTROLLER SIMULATION — BEST_k80 predictor (no switching cost)")
    print("=" * 70)

    df = load_predictions()
    all_data = pd.concat([df[df["split"] == "within"], df[df["split"] == "cross"]],
                          ignore_index=True)
    train_df, test_df = train_test_split(all_data)
    print(f"  Train={len(train_df)}  Test={len(test_df)}")

    rows = []
    for T in THRESHOLDS:
        print(f"\n{'─'*70}\n  THRESHOLD = {T}\n{'─'*70}", flush=True)
        sol, _ = get_solvable(test_df, T)
        print(f"  Solvable: {int(sol.sum())}/{len(test_df)}", flush=True)

        for name, m in [("AlwaysNano", 0), ("AlwaysSmall", 1), ("AlwaysMedium", 2)]:
            met = evaluate(test_df, always(test_df, m), T)
            met["threshold"] = T; met["controller"] = name
            rows.append(met)
            print(f"    {name:22s} sav={met['energy_savings_pct']:5.1f}% "
                  f"adq={met['adequate_rate']:5.1f}%", flush=True)

        met = evaluate(test_df, oracle_ctrl(test_df, T), T)
        met["threshold"] = T; met["controller"] = "Oracle"
        rows.append(met)
        print(f"    Oracle                 sav={met['energy_savings_pct']:5.1f}% "
              f"adq={met['adequate_rate']:5.1f}%", flush=True)

        print("    Optimizing DirectThreshold...", flush=True)
        mn, ms = optimize_direct(train_df, T)
        met = evaluate(test_df, direct_threshold(test_df, T, mn, ms), T)
        met["threshold"] = T
        met["controller"] = f"DirectThreshold(mn={mn:.2f},ms={ms:.2f})"
        rows.append(met)
        print(f"    DirectThreshold        sav={met['energy_savings_pct']:5.1f}% "
              f"adq={met['adequate_rate']:5.1f}%  mn={mn:.2f} ms={ms:.2f}", flush=True)

        print("    Optimizing BayesMPC...", flush=True)
        bcfg = optimize_bayes(train_df, T)
        met = evaluate(test_df, bayes_mpc(test_df, bcfg["H"], bcfg["lambda_u"], T), T)
        met["threshold"] = T
        met["controller"] = f"BayesMPC(H={bcfg['H']},lu={bcfg['lambda_u']})"
        rows.append(met)
        print(f"    BayesMPC               sav={met['energy_savings_pct']:5.1f}% "
              f"adq={met['adequate_rate']:5.1f}%  cfg={bcfg}", flush=True)

        print("    Optimizing RichMPC...", flush=True)
        rcfg = optimize_rich(train_df, T)
        met = evaluate(test_df, rich_mpc(test_df, rcfg["lambda_u"], rcfg["lambda_o"], T), T)
        met["threshold"] = T
        met["controller"] = f"RichMPC(lu={rcfg['lambda_u']},lo={rcfg['lambda_o']})"
        rows.append(met)
        print(f"    RichMPC                sav={met['energy_savings_pct']:5.1f}% "
              f"adq={met['adequate_rate']:5.1f}%  cfg={rcfg}", flush=True)

    cols = ["threshold", "n_solvable", "n_total", "controller",
            "energy_savings_pct", "adequate_rate", "correct_rate",
            "over_provision_rate", "mean_miss_rate",
            "pct_nano", "pct_small", "pct_medium", "switches_per_100"]
    out = pd.DataFrame(rows)[cols]
    out_tsv = SCRIPT_DIR / "controller_k80_results.tsv"
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n  Saved {out_tsv}", flush=True)

    # ── Comparison with SPATIAL_65 baseline ──
    base = pd.read_csv(BASELINE_TSV, sep="\t")

    def find(df_src, T, prefix):
        sub = df_src[(df_src["threshold"] == T) &
                      (df_src["controller"].str.startswith(prefix))]
        if len(sub) == 0: return None
        return sub.iloc[0]

    print("\n" + "=" * 88)
    print("  COMPARISON: SPATIAL_65 vs BEST_k80  (savings% / adequate%)")
    print("=" * 88)
    for ctrl in ["DirectThreshold", "BayesMPC", "RichMPC"]:
        print(f"\n  {ctrl}:")
        print(f"    {'Thresh':>6s} | {'SPATIAL_65':>16s} | {'BEST_k80':>16s} | "
              f"{'ΔSav':>7s} | {'ΔAdq':>7s}")
        for T in THRESHOLDS:
            b = find(base, T, ctrl)
            n = find(out, T, ctrl)
            if b is None or n is None: continue
            ds = n["energy_savings_pct"] - b["energy_savings_pct"]
            da = n["adequate_rate"] - b["adequate_rate"]
            flag = ""
            if da >= 1 and ds >= -1: flag = "  ↑ better"
            elif da <= -1 and ds <= 1: flag = "  ↓ worse"
            print(f"    {T:>6.2f} | "
                  f"{b['energy_savings_pct']:5.1f}%/{b['adequate_rate']:5.1f}%     | "
                  f"{n['energy_savings_pct']:5.1f}%/{n['adequate_rate']:5.1f}%     | "
                  f"{ds:+6.2f} | {da:+6.2f}{flag}")

    # ── Pareto plot ──
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(5 * len(THRESHOLDS), 5),
                              sharey=True)
    if len(THRESHOLDS) == 1:
        axes = [axes]
    for ax, T in zip(axes, THRESHOLDS):
        tr = [r for r in rows if r["threshold"] == T]
        for r in tr:
            name = r["controller"]
            if name == "Oracle":
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                           c="black", s=180, marker="*", zorder=6, label="Oracle")
            elif name == "AlwaysNano":
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                           c="green", s=100, marker="D", zorder=5, label="AlwaysNano")
            elif name == "AlwaysSmall":
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                           c="orange", s=100, marker="D", zorder=5, label="AlwaysSmall")
            elif name == "AlwaysMedium":
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                           c="purple", s=100, marker="D", zorder=5, label="AlwaysMedium")
            elif name.startswith("DirectThreshold"):
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                           c="limegreen", s=140, marker="P", zorder=6, label="DirectThreshold")
            elif name.startswith("BayesMPC"):
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                           c="red", s=120, marker="^", zorder=6, label="BayesMPC")
            elif name.startswith("RichMPC"):
                ax.scatter(r["energy_savings_pct"], r["adequate_rate"],
                           c="magenta", s=140, marker="v", zorder=6, label="RichMPC")
        n_sol = tr[0]["n_solvable"]; n_t = tr[0]["n_total"]
        ax.set_title(f"T={T} ({n_sol}/{n_t} solvable)", fontsize=11)
        ax.set_xlabel("Energy Savings (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
    axes[0].set_ylabel("Adequate Rate (%, solvable only)")
    axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plot = SCRIPT_DIR / "pareto_k80.png"
    plt.savefig(plot, dpi=150)
    print(f"\n  Saved {plot}", flush=True)
    print(f"\n  Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
