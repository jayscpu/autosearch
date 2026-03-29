"""
Main entry point for the ECHO controller simulation framework.

Runs all baselines and adaptive controllers, performs grid searches,
generates comparison tables and plots, and saves results to JSON.

Usage:
    python run_all.py --predictions ../second_autosearch/results/predictions_plainlstm_4cam.csv
    python run_all.py --all
    python run_all.py --all --skip-dqn
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

from controller.models import MODELS, energy_per_window
from controller.controllers import (
    AlwaysNano, AlwaysMedium, BestFixed, RandomController, OracleController,
    ThresholdController, ThresholdHysteresisController, BayesRiskMPCController,
    DQNController, ProxyController,
)
from controller.metrics import evaluate
from controller.search import (
    search_threshold, search_threshold_hysteresis, search_mpc,
    search_dqn, search_proxy,
)
from controller.budget import run_budget_constrained
from controller.visualize import (
    plot_timeline, plot_energy_comparison, plot_per_intersection,
)


PREDICTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "second_autosearch", "results")

ALL_PREDICTION_FILES = sorted(glob.glob(os.path.join(PREDICTION_DIR, "predictions_*.csv")))


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


def run_baselines(data: dict, t1: float, t2: float) -> list:
    """Run all 5 baseline controllers and return results."""
    pred = data["pred_miss_rates"]
    true = data["true_miss_rates"]
    unc = data["epistemic_unc"]

    baselines = [
        AlwaysNano(),
        AlwaysMedium(),
        BestFixed(model_idx=1),
        RandomController(seed=42),
        OracleController(t1=t1, t2=t2),
    ]

    results = []
    for ctrl in baselines:
        m = evaluate(ctrl, pred, true, MODELS, t1, t2, unc)
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


def run_per_intersection(data: dict, controllers: list, t1: float, t2: float) -> dict:
    """Run controllers per-intersection for 4CAM datasets.

    Returns:
        {intersection_name: {controller_name: metrics_dict}}
    """
    if data["intersections"] is None:
        return {}

    unique_ints = np.unique(data["intersections"])
    if len(unique_ints) <= 1:
        return {}

    print(f"\n── Per-Intersection Breakdown ({len(unique_ints)} intersections) ──")
    inter_results = {}

    for inter_name in unique_ints:
        mask = data["intersections"] == inter_name
        sub_pred = data["pred_miss_rates"][mask]
        sub_true = data["true_miss_rates"][mask]
        sub_unc = data["epistemic_unc"][mask] if data["epistemic_unc"] is not None else None

        inter_results[inter_name] = {}
        for ctrl in controllers:
            ctrl.reset()
            m = evaluate(ctrl, sub_pred, sub_true, MODELS, t1, t2, sub_unc)
            inter_results[inter_name][ctrl.name()] = m

    # Print summary
    cross_cam = "Bellevue_150th_SE38th"
    for inter_name in unique_ints:
        marker = " ★" if cross_cam in inter_name else ""
        n_frames = np.sum(data["intersections"] == inter_name)
        print(f"  {inter_name}{marker} (n={n_frames}):")
        for ctrl_name, m in inter_results[inter_name].items():
            print(f"    {ctrl_name:30s} | savings={m['energy_savings_pct']:.1f}% | "
                  f"adequate={m['adequate_rate']:.3f}")

    return inter_results


def process_prediction_file(csv_path: str, skip_dqn: bool = False):
    """Process a single prediction file: baselines, searches, budget, plots."""
    print(f"\n{'#'*100}")
    print(f"# Processing: {os.path.basename(csv_path)}")
    print(f"{'#'*100}")

    data = load_predictions(csv_path)
    pred = data["pred_miss_rates"]
    true = data["true_miss_rates"]
    unc = data["epistemic_unc"]
    file_tag = data["name"]

    has_unc = unc is not None
    is_4cam = "4cam" in file_tag.lower()

    print(f"  Frames: {data['n']} | Has uncertainty: {has_unc} | 4CAM: {is_4cam}")

    all_results = []

    # ── 1. Baselines ─────────────────────────────────────────────────────
    print("\n── Baselines ──")
    baseline_results = run_baselines(data, t1=0.15, t2=0.35)
    all_results.extend(baseline_results)

    # Get oracle energy for reference
    oracle_energy = None
    for r in baseline_results:
        if "Oracle" in r["controller"]:
            oracle_energy = r["avg_energy_mj"]
            break

    # ── 2. Threshold search ──────────────────────────────────────────────
    print("\n── Threshold Grid Search ──")
    thresh_cfg, thresh_best, thresh_ranked = search_threshold(pred, true, unc)
    t1_best, t2_best = thresh_cfg["t1"], thresh_cfg["t2"]
    all_results.append(thresh_best)

    # ── 3. Hysteresis search ─────────────────────────────────────────────
    print("\n── Hysteresis Grid Search ──")
    hyst_cfg, hyst_best, hyst_ranked = search_threshold_hysteresis(pred, true, unc)
    all_results.append(hyst_best)

    # ── 4. MPC search (warm-started) ─────────────────────────────────────
    print("\n── MPC Grid Search (warm-started) ──")
    mpc_cfg, mpc_best, mpc_ranked = search_mpc(
        pred, true, threshold_results=thresh_ranked, epistemic_unc=unc)
    all_results.append(mpc_best)

    # ── 5. DQN search ────────────────────────────────────────────────────
    dqn_best_result = None
    if not skip_dqn:
        print("\n── DQN Grid Search ──")
        try:
            import torch  # noqa: F401
            dqn_cfg, dqn_best_result, dqn_ranked = search_dqn(
                pred, true, t1=t1_best, t2=t2_best, epistemic_unc=unc)
            if dqn_best_result:
                all_results.append(dqn_best_result)
        except ImportError:
            print("  PyTorch not available — skipping DQN search.")
    else:
        print("\n── DQN search skipped (--skip-dqn) ──")

    # ── 6. Proxy search ──────────────────────────────────────────────────
    proxy_best_result = None
    if has_unc:
        print("\n── Proxy (Uncertainty) Search ──")
        proxy_cfg, proxy_best_result, proxy_ranked = search_proxy(
            pred, true, unc, t1=t1_best, t2=t2_best)
        all_results.append(proxy_best_result)
    else:
        print("\n── Proxy search skipped (no epistemic_unc column) ──")

    # ── 7. Budget-constrained runs ───────────────────────────────────────
    print("\n── Budget-Constrained Simulation ──")
    # Budget = enough for always-small
    n_frames = data["n"]
    small_total_energy_j = n_frames * energy_per_window(1) / 1000.0
    print(f"  Budget = {small_total_energy_j:.2f} J (equivalent to always-small)")

    # Run with best threshold controller
    best_thresh_ctrl = ThresholdController(t1=t1_best, t2=t2_best)
    budget_thresh = run_budget_constrained(
        best_thresh_ctrl, pred, true, MODELS, t1_best, t2_best,
        total_budget_j=small_total_energy_j, epistemic_unc=unc)
    print(f"  Threshold [budget]: degraded={budget_thresh['time_in_degraded_pct']:.1f}%, "
          f"utilization={budget_thresh['final_budget_utilization']:.3f}")

    # Run with best MPC controller
    best_mpc_ctrl = BayesRiskMPCController(
        horizon=mpc_cfg["horizon"], lambda_under=mpc_cfg["lambda_under"],
        lambda_over=mpc_cfg["lambda_over"], w_switch=mpc_cfg["w_switch"],
        t1=mpc_cfg["t1"], t2=mpc_cfg["t2"])
    budget_mpc = run_budget_constrained(
        best_mpc_ctrl, pred, true, MODELS,
        mpc_cfg["t1"], mpc_cfg["t2"],
        total_budget_j=small_total_energy_j, epistemic_unc=unc)
    print(f"  MPC [budget]:       degraded={budget_mpc['time_in_degraded_pct']:.1f}%, "
          f"utilization={budget_mpc['final_budget_utilization']:.3f}")

    all_results.extend([budget_thresh, budget_mpc])

    # ── 8. Comparison table ──────────────────────────────────────────────
    print_comparison_table(all_results, title=f"Results for {file_tag}")

    # ── 9. Per-intersection breakdown (4CAM only) ────────────────────────
    inter_results = {}
    if is_4cam:
        top_ctrls = [
            ThresholdController(t1=t1_best, t2=t2_best),
            BayesRiskMPCController(
                horizon=mpc_cfg["horizon"], lambda_under=mpc_cfg["lambda_under"],
                lambda_over=mpc_cfg["lambda_over"], w_switch=mpc_cfg["w_switch"],
                t1=mpc_cfg["t1"], t2=mpc_cfg["t2"]),
            OracleController(t1=t1_best, t2=t2_best),
        ]
        inter_results = run_per_intersection(data, top_ctrls, t1_best, t2_best)

    # ── 10. Plots ────────────────────────────────────────────────────────
    print("\n── Generating Plots ──")

    # Timeline plot
    plot_timeline(
        pred, true, all_results,
        threshold_name=thresh_best["controller"],
        mpc_name=mpc_best["controller"],
        t1=t1_best, t2=t2_best,
        predictor_name=file_tag,
        save_name=f"timeline_{file_tag}",
    )

    # Energy comparison bar chart
    medium_energy_val = energy_per_window(2)
    plot_energy_comparison(
        all_results,
        oracle_energy=oracle_energy,
        medium_energy=medium_energy_val,
        save_name=f"energy_{file_tag}",
    )

    # Per-intersection plot (4CAM only)
    if inter_results:
        top_ctrl_names = [c.name() for c in top_ctrls[:3]]
        plot_per_intersection(
            inter_results, top_ctrl_names,
            save_name=f"intersection_{file_tag}",
        )

    # ── 11. Save results JSON ────────────────────────────────────────────
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items()
              if k not in ("selections", "energies", "oracle_selections")}
        # Convert numpy types to Python types for JSON serialization
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
        "best_mpc": mpc_cfg,
        "inter_results": inter_results,
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

    all_outputs = []
    for csv_path in files:
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        output = process_prediction_file(csv_path, skip_dqn=args.skip_dqn)
        all_outputs.append(output)

    print(f"\n{'='*100}")
    print(f"  DONE — processed {len(all_outputs)} prediction file(s)")
    print(f"  Plots saved to: controller/plots/")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
