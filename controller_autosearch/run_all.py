"""
Main entry point for the ECHO controller simulation framework.

Runs all baselines and adaptive controllers, performs grid searches,
generates comparison tables and plots, and saves results to JSON.

IMPORTANT: All grid searches optimize on the first 60% of data (train split).
All reported metrics in the comparison table come from the last 40% (test split).
This ensures all rows are comparable.

Usage:
    python run_all.py --predictions ../second_autosearch/results/predictions_plainlstm_4cam.csv
    python run_all.py --all
    python run_all.py --all --skip-dqn
    python run_all.py --all --t1 0.10 --t2 0.70
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller_autosearch.models import MODELS, energy_per_window
from controller_autosearch.controllers import (
    AlwaysNano, AlwaysMedium, BestFixed, RandomController, OracleController,
    ThresholdController, ThresholdHysteresisController, BayesRiskMPCController,
    DQNController, ProxyController,
)
from controller_autosearch.metrics import evaluate
from controller_autosearch.search import (
    search_threshold, search_threshold_hysteresis, search_mpc,
    search_dqn, search_proxy, pareto_sweep,
)
from controller_autosearch.budget import run_budget_constrained
from controller_autosearch.visualize import (
    plot_timeline, plot_energy_comparison, plot_per_intersection,
    plot_pareto_frontier,
)


PREDICTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "second_autosearch", "results")

ALL_PREDICTION_FILES = sorted(glob.glob(os.path.join(PREDICTION_DIR, "predictions_*.csv")))

TRAIN_FRAC = 0.60  # first 60% for optimization, last 40% for reporting


def load_predictions(csv_path: str) -> dict:
    """Load a prediction CSV and return arrays + metadata."""
    df = pd.read_csv(csv_path)
    has_unc = "epistemic_unc" in df.columns
    has_intersection = "intersection" in df.columns

    data = {
        "path": csv_path,
        "name": os.path.splitext(os.path.basename(csv_path))[0],
        "pred_miss_rates": df["pred_miss_rate"].values,
        "true_miss_rates": df["true_miss_rate"].values,
        "epistemic_unc": df["epistemic_unc"].values if has_unc else None,
        "intersections": df["intersection"].values if has_intersection else None,
        "df": df,
        "n": len(df),
    }
    return data


def split_data(data: dict, frac: float = TRAIN_FRAC) -> tuple:
    """Split prediction data into train/test portions.

    Returns:
        (train_dict, test_dict) with same keys as data but sliced arrays.
    """
    n = data["n"]
    split_idx = int(frac * n)

    def _slice(arr, start, end):
        return arr[start:end] if arr is not None else None

    train = {
        "pred_miss_rates": data["pred_miss_rates"][:split_idx],
        "true_miss_rates": data["true_miss_rates"][:split_idx],
        "epistemic_unc": _slice(data["epistemic_unc"], 0, split_idx),
        "intersections": _slice(data["intersections"], 0, split_idx),
        "n": split_idx,
    }
    test = {
        "pred_miss_rates": data["pred_miss_rates"][split_idx:],
        "true_miss_rates": data["true_miss_rates"][split_idx:],
        "epistemic_unc": _slice(data["epistemic_unc"], split_idx, n),
        "intersections": _slice(data["intersections"], split_idx, n),
        "n": n - split_idx,
    }
    return train, test


def evaluate_on_test(ctrl, test: dict, t1_eval: float, t2_eval: float) -> dict:
    """Evaluate a controller on the test split with fixed eval thresholds."""
    return evaluate(ctrl, test["pred_miss_rates"], test["true_miss_rates"],
                    MODELS, t1_eval, t2_eval, test["epistemic_unc"])


def run_baselines_on_test(test: dict, t1_eval: float, t2_eval: float) -> list:
    """Run all 5 baseline controllers on the test split."""
    baselines = [
        AlwaysNano(),
        AlwaysMedium(),
        BestFixed(model_idx=1),
        RandomController(seed=42),
        OracleController(t1=t1_eval, t2=t2_eval),
    ]

    results = []
    for ctrl in baselines:
        m = evaluate_on_test(ctrl, test, t1_eval, t2_eval)
        results.append(m)
        print(f"  {ctrl.name():30s} | energy={m['avg_energy_mj']:.1f}mJ | "
              f"savings={m['energy_savings_pct']:.1f}% | "
              f"adequate={m['adequate_rate']:.3f}")
    return results


def print_comparison_table(results: list, title: str = ""):
    """Print a formatted comparison table of controller results."""
    print(f"\n{'='*100}")
    if title:
        print(f"  {title}")
        print(f"{'='*100}")

    header = (f"{'Controller':35s} | {'Energy(mJ)':>10s} | {'Savings%':>8s} | "
              f"{'OracGap':>7s} | {'Adequate':>8s} | {'Sw/100':>6s} | "
              f"{'Nano%':>5s} {'Sm%':>5s} {'Med%':>5s} | {'Lat(ms)':>7s}")
    print(header)
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["avg_energy_mj"]):
        dist = r["model_distribution"]
        oracle_gap = r.get("oracle_gap", float("nan"))
        print(f"{r['controller']:35s} | {r['avg_energy_mj']:10.1f} | "
              f"{r['energy_savings_pct']:7.1f}% | "
              f"{oracle_gap:7.3f} | {r['adequate_rate']:8.3f} | "
              f"{r['switches_per_100']:6.1f} | "
              f"{dist.get('nano', 0):5.1f} {dist.get('small', 0):5.1f} "
              f"{dist.get('medium', 0):5.1f} | {r['avg_latency_ms']:7.2f}")

    print("=" * 100)


def run_per_intersection(test: dict, controllers: list,
                         t1_eval: float, t2_eval: float) -> dict:
    """Run controllers per-intersection on the test split (4CAM only).

    Returns:
        {intersection_name: {controller_name: metrics_dict}}
    """
    if test["intersections"] is None:
        return {}

    unique_ints = np.unique(test["intersections"])
    if len(unique_ints) <= 1:
        return {}

    print(f"\n── Per-Intersection Breakdown ({len(unique_ints)} intersections) ──")
    inter_results = {}

    for inter_name in unique_ints:
        mask = test["intersections"] == inter_name
        sub_pred = test["pred_miss_rates"][mask]
        sub_true = test["true_miss_rates"][mask]
        sub_unc = test["epistemic_unc"][mask] if test["epistemic_unc"] is not None else None

        inter_results[inter_name] = {}
        for ctrl in controllers:
            ctrl.reset()
            m = evaluate(ctrl, sub_pred, sub_true, MODELS, t1_eval, t2_eval, sub_unc)
            inter_results[inter_name][ctrl.name()] = m

    # Print summary
    cross_cam = "Bellevue_150th_SE38th"
    for inter_name in unique_ints:
        marker = " ★" if cross_cam in inter_name else ""
        n_frames = np.sum(test["intersections"] == inter_name)
        print(f"  {inter_name}{marker} (n={n_frames}):")
        for ctrl_name, m in inter_results[inter_name].items():
            print(f"    {ctrl_name:30s} | savings={m['energy_savings_pct']:.1f}% | "
                  f"adequate={m['adequate_rate']:.3f}")

    return inter_results


def process_prediction_file(csv_path: str, skip_dqn: bool = False,
                            default_t1: float = 0.15, default_t2: float = 0.35):
    """Process a single prediction file: baselines, searches, budget, plots.

    All grid searches optimize on the train split (first 60%).
    All reported metrics come from the test split (last 40%).
    """
    print(f"\n{'#'*100}")
    print(f"# Processing: {os.path.basename(csv_path)}")
    print(f"{'#'*100}")

    data = load_predictions(csv_path)
    file_tag = data["name"]
    t1_eval, t2_eval = default_t1, default_t2

    has_unc = data["epistemic_unc"] is not None
    is_4cam = "4cam" in file_tag.lower()

    # ── Train/test split ─────────────────────────────────────────────────
    train, test = split_data(data, TRAIN_FRAC)
    print(f"  Total frames: {data['n']} | Train: {train['n']} | Test: {test['n']}")
    print(f"  Has uncertainty: {has_unc} | 4CAM: {is_4cam}")
    print(f"  Eval thresholds: t1_eval={t1_eval}, t2_eval={t2_eval}")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 1: Grid searches on TRAIN split
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PHASE 1: Grid searches on train split ({train['n']} frames)")
    print(f"{'─'*60}")

    # ── 1. Threshold search ──────────────────────────────────────────────
    print("\n── Threshold Grid Search ──")
    thresh_cfg, _, thresh_ranked = search_threshold(
        train["pred_miss_rates"], train["true_miss_rates"],
        t1_eval, t2_eval, train["epistemic_unc"])
    t1_best, t2_best = thresh_cfg["t1"], thresh_cfg["t2"]

    # ── 2. Hysteresis search ─────────────────────────────────────────────
    print("\n── Hysteresis Grid Search ──")
    hyst_cfg, _, hyst_ranked = search_threshold_hysteresis(
        train["pred_miss_rates"], train["true_miss_rates"],
        t1_eval, t2_eval, train["epistemic_unc"])

    # ── 3. MPC search (warm-started) ─────────────────────────────────────
    print("\n── MPC Grid Search (warm-started) ──")
    mpc_cfg, _, mpc_ranked = search_mpc(
        train["pred_miss_rates"], train["true_miss_rates"],
        t1_eval, t2_eval,
        threshold_results=thresh_ranked, epistemic_unc=train["epistemic_unc"])

    # ── 4. DQN search ────────────────────────────────────────────────────
    dqn_cfg = None
    if not skip_dqn:
        print("\n── DQN Grid Search ──")
        try:
            import torch  # noqa: F401
            dqn_cfg, _, dqn_ranked = search_dqn(
                train["pred_miss_rates"], train["true_miss_rates"],
                t1_eval, t2_eval, epistemic_unc=train["epistemic_unc"])
        except ImportError:
            print("  PyTorch not available — skipping DQN search.")
    else:
        print("\n── DQN search skipped (--skip-dqn) ──")

    # ── 5. Proxy search ──────────────────────────────────────────────────
    proxy_cfg = None
    if has_unc:
        print("\n── Proxy (Uncertainty) Search ──")
        proxy_cfg, _, proxy_ranked = search_proxy(
            train["pred_miss_rates"], train["true_miss_rates"],
            train["epistemic_unc"], t1_eval, t2_eval)
    else:
        print("\n── Proxy search skipped (no epistemic_unc column) ──")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 2: Evaluate ALL controllers on TEST split
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PHASE 2: Evaluate on test split ({test['n']} frames)")
    print(f"{'─'*60}")

    all_results = []

    # ── Baselines on test ────────────────────────────────────────────────
    print("\n── Baselines (test) ──")
    baseline_results = run_baselines_on_test(test, t1_eval, t2_eval)
    all_results.extend(baseline_results)

    oracle_energy = None
    for r in baseline_results:
        if "Oracle" in r["controller"]:
            oracle_energy = r["avg_energy_mj"]
            break

    # ── Best threshold on test ───────────────────────────────────────────
    print("\n── Best Threshold (test) ──")
    best_thresh_ctrl = ThresholdController(t1=t1_best, t2=t2_best)
    thresh_test = evaluate_on_test(best_thresh_ctrl, test, t1_eval, t2_eval)
    print(f"  {thresh_test['controller']:30s} | energy={thresh_test['avg_energy_mj']:.1f}mJ | "
          f"adequate={thresh_test['adequate_rate']:.3f}")
    all_results.append(thresh_test)

    # ── Best hysteresis on test ──────────────────────────────────────────
    print("\n── Best Hysteresis (test) ──")
    best_hyst_ctrl = ThresholdHysteresisController(
        t1=hyst_cfg["t1"], t2=hyst_cfg["t2"],
        hysteresis_n=hyst_cfg["hysteresis_n"])
    hyst_test = evaluate_on_test(best_hyst_ctrl, test, t1_eval, t2_eval)
    print(f"  {hyst_test['controller']:30s} | energy={hyst_test['avg_energy_mj']:.1f}mJ | "
          f"adequate={hyst_test['adequate_rate']:.3f}")
    all_results.append(hyst_test)

    # ── Best MPC on test ─────────────────────────────────────────────────
    print("\n── Best MPC (test) ──")
    best_mpc_ctrl = BayesRiskMPCController(
        horizon=mpc_cfg["horizon"], lambda_under=mpc_cfg["lambda_under"],
        lambda_over=mpc_cfg["lambda_over"], w_switch=mpc_cfg["w_switch"],
        t1=mpc_cfg["t1"], t2=mpc_cfg["t2"])
    mpc_test = evaluate_on_test(best_mpc_ctrl, test, t1_eval, t2_eval)
    print(f"  {mpc_test['controller']:30s} | energy={mpc_test['avg_energy_mj']:.1f}mJ | "
          f"adequate={mpc_test['adequate_rate']:.3f}")
    all_results.append(mpc_test)

    # ── Best DQN on test (retrain on full train data) ────────────────────
    if dqn_cfg is not None:
        print("\n── Best DQN (retrain on train, eval on test) ──")
        best_dqn_ctrl = DQNController(
            lambda_tradeoff=dqn_cfg["lambda_tradeoff"],
            alpha_switch=dqn_cfg["alpha_switch"],
            lr=dqn_cfg["lr"])
        best_dqn_ctrl.train(train["pred_miss_rates"], train["true_miss_rates"],
                            t1_eval, t2_eval,
                            n_episodes=dqn_cfg["n_train_episodes"])
        dqn_test = evaluate_on_test(best_dqn_ctrl, test, t1_eval, t2_eval)
        print(f"  {dqn_test['controller']:30s} | energy={dqn_test['avg_energy_mj']:.1f}mJ | "
              f"adequate={dqn_test['adequate_rate']:.3f}")
        all_results.append(dqn_test)

    # ── Best proxy on test ───────────────────────────────────────────────
    if proxy_cfg is not None:
        print("\n── Best Proxy (test) ──")
        best_proxy_ctrl = ProxyController(
            unc_threshold_low=proxy_cfg["unc_threshold_low"],
            unc_threshold_high=proxy_cfg["unc_threshold_high"])
        best_proxy_ctrl.set_uncertainty(test["epistemic_unc"])
        proxy_test = evaluate_on_test(best_proxy_ctrl, test, t1_eval, t2_eval)
        print(f"  {proxy_test['controller']:30s} | energy={proxy_test['avg_energy_mj']:.1f}mJ | "
              f"adequate={proxy_test['adequate_rate']:.3f}")
        all_results.append(proxy_test)

    # ── Pareto frontier sweep (on test data) ─────────────────────────────
    print("\n── Pareto Frontier Sweep (test) ──")
    pareto_results = pareto_sweep(
        test["pred_miss_rates"], test["true_miss_rates"],
        t1_eval, t2_eval, MODELS, epistemic_unc=test["epistemic_unc"])

    print(f"\n  {'Constraint':>12s} | {'Savings%':>9s} | {'Adequate':>9s} | "
          f"{'t1':>6s} | {'t2':>6s}")
    print("  " + "-" * 55)
    for constraint, savings, adequate, cfg in pareto_results:
        print(f"  {constraint:>11.2f}  | {savings:>8.1f}% | {adequate:>9.3f} | "
              f"{cfg['t1']:>6.3f} | {cfg['t2']:>6.3f}")

    pareto_plot_path = os.path.join(os.path.dirname(__file__), "plots",
                                    f"pareto_{file_tag}.png")
    plot_pareto_frontier(pareto_results, save_path=pareto_plot_path)

    # ── Budget-constrained runs (on test data, with kappa grid) ──────────
    print("\n── Budget-Constrained Simulation (test) ──")
    n_test = test["n"]
    small_total_energy_j = n_test * energy_per_window(1) / 1000.0
    print(f"  Budget = {small_total_energy_j:.2f} J (equivalent to always-small over test)")

    kappa_vals = [1.0, 1.5, 2.0, 3.0]

    for kappa in kappa_vals:
        # Threshold + budget
        ctrl_t = ThresholdController(t1=t1_best, t2=t2_best)
        bt = run_budget_constrained(
            ctrl_t, test["pred_miss_rates"], test["true_miss_rates"],
            MODELS, t1_eval, t2_eval,
            total_budget_j=small_total_energy_j, kappa=kappa,
            epistemic_unc=test["epistemic_unc"])
        print(f"  Threshold [k={kappa}]: degraded={bt['time_in_degraded_pct']:.1f}%, "
              f"util={bt['final_budget_utilization']:.3f}, "
              f"adequate={bt['adequate_rate']:.3f}")
        all_results.append(bt)

        # MPC + budget
        ctrl_m = BayesRiskMPCController(
            horizon=mpc_cfg["horizon"], lambda_under=mpc_cfg["lambda_under"],
            lambda_over=mpc_cfg["lambda_over"], w_switch=mpc_cfg["w_switch"],
            t1=mpc_cfg["t1"], t2=mpc_cfg["t2"])
        bm = run_budget_constrained(
            ctrl_m, test["pred_miss_rates"], test["true_miss_rates"],
            MODELS, t1_eval, t2_eval,
            total_budget_j=small_total_energy_j, kappa=kappa,
            epistemic_unc=test["epistemic_unc"])
        print(f"  MPC       [k={kappa}]: degraded={bm['time_in_degraded_pct']:.1f}%, "
              f"util={bm['final_budget_utilization']:.3f}, "
              f"adequate={bm['adequate_rate']:.3f}")
        all_results.append(bm)

    # ── Comparison table (all test-set metrics) ──────────────────────────
    print_comparison_table(all_results, title=f"Results for {file_tag} (TEST split, n={n_test})")

    # ── Per-intersection breakdown (4CAM only, test split) ───────────────
    inter_results = {}
    if is_4cam:
        top_ctrls = [
            ThresholdController(t1=t1_best, t2=t2_best),
            BayesRiskMPCController(
                horizon=mpc_cfg["horizon"], lambda_under=mpc_cfg["lambda_under"],
                lambda_over=mpc_cfg["lambda_over"], w_switch=mpc_cfg["w_switch"],
                t1=mpc_cfg["t1"], t2=mpc_cfg["t2"]),
            OracleController(t1=t1_eval, t2=t2_eval),
        ]
        inter_results = run_per_intersection(test, top_ctrls, t1_eval, t2_eval)

    # ── Plots ────────────────────────────────────────────────────────────
    print("\n── Generating Plots ──")

    plot_timeline(
        test["pred_miss_rates"], test["true_miss_rates"], all_results,
        threshold_name=thresh_test["controller"],
        mpc_name=mpc_test["controller"],
        t1=t1_eval, t2=t2_eval,
        predictor_name=file_tag,
        save_name=f"timeline_{file_tag}",
    )

    medium_energy_val = energy_per_window(2)
    plot_energy_comparison(
        all_results,
        oracle_energy=oracle_energy,
        medium_energy=medium_energy_val,
        save_name=f"energy_{file_tag}",
    )

    if inter_results:
        top_ctrl_names = [c.name() for c in top_ctrls[:3]]
        plot_per_intersection(
            inter_results, top_ctrl_names,
            save_name=f"intersection_{file_tag}",
        )

    # ── Save results JSON ────────────────────────────────────────────────
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items()
              if k not in ("selections", "energies", "oracle_selections")}
        for key in jr:
            if isinstance(jr[key], np.floating):
                jr[key] = float(jr[key])
            elif isinstance(jr[key], np.integer):
                jr[key] = int(jr[key])
            elif isinstance(jr[key], dict):
                jr[key] = {k: float(v) if isinstance(v, np.floating) else v
                           for k, v in jr[key].items()}
        json_results.append(jr)

    json_path = os.path.join(os.path.dirname(__file__), f"results_{file_tag}.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    return {
        "file": file_tag,
        "all_results": all_results,
        "best_threshold": thresh_cfg,
        "best_hysteresis": hyst_cfg,
        "best_mpc": mpc_cfg,
        "best_dqn": dqn_cfg,
        "best_proxy": proxy_cfg,
        "inter_results": inter_results,
        "pareto": pareto_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ECHO Controller Simulation Framework")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to a single prediction CSV file")
    parser.add_argument("--all", action="store_true",
                        help="Run all prediction files in results directory")
    parser.add_argument("--skip-dqn", action="store_true",
                        help="Skip DQN grid search (requires PyTorch)")
    parser.add_argument("--t1", type=float, default=0.15,
                        help="Fixed t1_eval threshold for adequacy scoring (default: 0.15)")
    parser.add_argument("--t2", type=float, default=0.35,
                        help="Fixed t2_eval threshold for adequacy scoring (default: 0.35)")
    args = parser.parse_args()

    if args.all:
        files = ALL_PREDICTION_FILES
        if not files:
            print("No prediction files found in:", PREDICTION_DIR)
            sys.exit(1)
        print(f"Running all {len(files)} prediction files:")
        for f in files:
            print(f"  - {os.path.basename(f)}")
    elif args.predictions:
        files = [args.predictions]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nEval thresholds: t1={args.t1}, t2={args.t2}")
    print(f"Train/test split: {TRAIN_FRAC:.0%} / {1-TRAIN_FRAC:.0%}")

    all_outputs = []
    for csv_path in files:
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        output = process_prediction_file(csv_path, skip_dqn=args.skip_dqn,
                                         default_t1=args.t1, default_t2=args.t2)
        all_outputs.append(output)

    print(f"\n{'='*100}")
    print(f"  DONE — processed {len(all_outputs)} prediction file(s)")
    print(f"  Plots saved to: controller_autosearch/plots/")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
