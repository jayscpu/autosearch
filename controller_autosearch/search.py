"""
Grid search functions for optimizing controller parameters.

Each search function runs all parameter combinations and returns the best
configuration that minimizes energy subject to adequate_rate >= 0.85.

IMPORTANT: All search functions take t1_eval/t2_eval — fixed evaluation
thresholds that define what "adequate" means. These are passed to evaluate()
for scoring. The controller's own internal thresholds are grid-searched
separately and do NOT affect adequacy scoring.
"""

import numpy as np
from itertools import product

from .models import MODELS
from .controllers import (ThresholdController, ThresholdHysteresisController,
                          BayesRiskMPCController, DQNController, ProxyController)
from .metrics import evaluate


def _filter_and_rank(results: list, min_adequate: float = 0.85) -> list:
    """Filter results by adequacy constraint, then sort by energy (ascending)."""
    feasible = [r for r in results if r["adequate_rate"] >= min_adequate]
    if not feasible:
        # If no config meets the constraint, return all sorted by adequacy (desc)
        return sorted(results, key=lambda r: -r["adequate_rate"])
    return sorted(feasible, key=lambda r: r["avg_energy_mj"])


def _print_progress(current: int, total: int, name: str, interval: int = 10):
    """Print progress at every interval% milestone."""
    pct = int(current / total * 100)
    prev_pct = int((current - 1) / total * 100) if current > 0 else -1
    if pct // interval > prev_pct // interval or current == total - 1:
        print(f"  [{name}] {pct}% ({current+1}/{total})", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold search
# ═══════════════════════════════════════════════════════════════════════════════

def search_threshold(pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
                     t1_eval: float, t2_eval: float,
                     epistemic_unc: np.ndarray = None,
                     min_adequate: float = 0.85) -> tuple:
    """Grid search over threshold parameters t1 and t2.

    Grid: t1 in linspace(0.01, 0.50, 30), t2 in linspace(0.05, 0.80, 30),
    constrained t2 > t1. ~400 feasible combinations.

    Args:
        t1_eval, t2_eval: Fixed evaluation thresholds for adequacy scoring.

    Returns:
        (best_config, best_metrics, all_results_sorted)
    """
    t1_vals = np.linspace(0.01, 0.50, 30)
    t2_vals = np.linspace(0.05, 0.80, 30)

    combos = [(t1, t2) for t1 in t1_vals for t2 in t2_vals if t2 > t1]
    total = len(combos)
    print(f"[Threshold Search] {total} combinations")

    results = []
    for i, (t1, t2) in enumerate(combos):
        _print_progress(i, total, "Threshold")
        ctrl = ThresholdController(t1=t1, t2=t2)
        m = evaluate(ctrl, pred_miss_rates, true_miss_rates, MODELS,
                     t1_eval, t2_eval, epistemic_unc)
        m["config"] = {"t1": t1, "t2": t2}
        results.append(m)

    ranked = _filter_and_rank(results, min_adequate)
    best = ranked[0] if ranked else results[0]
    print(f"  Best: t1={best['config']['t1']:.3f}, t2={best['config']['t2']:.3f} | "
          f"energy={best['avg_energy_mj']:.1f}mJ | adequate={best['adequate_rate']:.3f}")
    return best["config"], best, ranked


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold + Hysteresis search
# ═══════════════════════════════════════════════════════════════════════════════

def search_threshold_hysteresis(pred_miss_rates: np.ndarray,
                                true_miss_rates: np.ndarray,
                                t1_eval: float, t2_eval: float,
                                epistemic_unc: np.ndarray = None,
                                min_adequate: float = 0.85) -> tuple:
    """Grid search over threshold + hysteresis parameters.

    Grid: same t1/t2 as threshold search, plus hysteresis_n in {2,3,5,8,10,15}.
    ~2400 combinations.

    Returns:
        (best_config, best_metrics, all_results_sorted)
    """
    t1_vals = np.linspace(0.01, 0.50, 30)
    t2_vals = np.linspace(0.05, 0.80, 30)
    hyst_vals = [2, 3, 5, 8, 10, 15]

    combos = [(t1, t2, h) for t1 in t1_vals for t2 in t2_vals
              for h in hyst_vals if t2 > t1]
    total = len(combos)
    print(f"[Hysteresis Search] {total} combinations")

    results = []
    for i, (t1, t2, h) in enumerate(combos):
        _print_progress(i, total, "Hysteresis")
        ctrl = ThresholdHysteresisController(t1=t1, t2=t2, hysteresis_n=h)
        m = evaluate(ctrl, pred_miss_rates, true_miss_rates, MODELS,
                     t1_eval, t2_eval, epistemic_unc)
        m["config"] = {"t1": t1, "t2": t2, "hysteresis_n": h}
        results.append(m)

    ranked = _filter_and_rank(results, min_adequate)
    best = ranked[0] if ranked else results[0]
    print(f"  Best: t1={best['config']['t1']:.3f}, t2={best['config']['t2']:.3f}, "
          f"n={best['config']['hysteresis_n']} | "
          f"energy={best['avg_energy_mj']:.1f}mJ | adequate={best['adequate_rate']:.3f}")
    return best["config"], best, ranked


# ═══════════════════════════════════════════════════════════════════════════════
# MPC search (warm-started from threshold results)
# ═══════════════════════════════════════════════════════════════════════════════

def search_mpc(pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
               t1_eval: float, t2_eval: float,
               threshold_results: list = None,
               epistemic_unc: np.ndarray = None,
               min_adequate: float = 0.85) -> tuple:
    """Grid search over MPC parameters, warm-started from best threshold configs.

    Takes the top 3 (t1, t2) pairs from threshold search to avoid redundant
    exploration. ~270 combos per (t1,t2) pair, ~810 total.

    Returns:
        (best_config, best_metrics, all_results_sorted)
    """
    horizons = [2, 3, 5]
    lambda_under_vals = [0.3, 0.5, 1.0, 2.0, 3.0]
    lambda_over_vals = [0.1, 0.3, 0.5, 1.0, 2.0]
    w_switch_vals = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1]

    # Warm-start: use top 3 threshold configs
    if threshold_results and len(threshold_results) >= 3:
        t_pairs = [(r["config"]["t1"], r["config"]["t2"])
                   for r in threshold_results[:3]]
    else:
        # Fallback defaults
        t_pairs = [(0.10, 0.30), (0.15, 0.35), (0.20, 0.40)]

    combos = [(h, lu, lo, ws, t1, t2)
              for t1, t2 in t_pairs
              for h in horizons
              for lu in lambda_under_vals
              for lo in lambda_over_vals
              for ws in w_switch_vals]
    total = len(combos)
    print(f"[MPC Search] {total} combinations")

    results = []
    for i, (h, lu, lo, ws, t1, t2) in enumerate(combos):
        _print_progress(i, total, "MPC")
        ctrl = BayesRiskMPCController(horizon=h, lambda_under=lu, lambda_over=lo,
                                      w_switch=ws, t1=t1, t2=t2)
        m = evaluate(ctrl, pred_miss_rates, true_miss_rates, MODELS,
                     t1_eval, t2_eval, epistemic_unc)
        m["config"] = {"horizon": h, "lambda_under": lu, "lambda_over": lo,
                       "w_switch": ws, "t1": t1, "t2": t2}
        results.append(m)

    ranked = _filter_and_rank(results, min_adequate)
    best = ranked[0] if ranked else results[0]
    print(f"  Best: H={best['config']['horizon']}, "
          f"λu={best['config']['lambda_under']}, λo={best['config']['lambda_over']}, "
          f"ws={best['config']['w_switch']} | "
          f"energy={best['avg_energy_mj']:.1f}mJ | adequate={best['adequate_rate']:.3f}")
    return best["config"], best, ranked


# ═══════════════════════════════════════════════════════════════════════════════
# DQN search
# ═══════════════════════════════════════════════════════════════════════════════

def search_dqn(pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
               t1_eval: float, t2_eval: float,
               epistemic_unc: np.ndarray = None,
               min_adequate: float = 0.85) -> tuple:
    """Grid search over DQN hyperparameters.

    Splits the provided data 60/40 internally for DQN train/validate.
    150 combinations (slowest search due to training).

    Args:
        pred_miss_rates: Training portion of predictions (from the global 60% split).
        true_miss_rates: Training portion of ground truth.
        t1_eval, t2_eval: Fixed evaluation thresholds for adequacy scoring.

    Returns:
        (best_config, best_metrics, all_results_sorted)
    """
    lambda_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    alpha_vals = [0.0, 0.005, 0.01, 0.03, 0.05]
    lr_vals = [5e-4, 1e-3, 2e-3]
    episode_vals = [10, 20]

    combos = list(product(lambda_vals, alpha_vals, lr_vals, episode_vals))
    total = len(combos)
    print(f"[DQN Search] {total} combinations")

    # DQN-specific internal split: train on first 60%, validate on last 40%
    # of the provided (already-split) training data
    split = int(0.6 * len(pred_miss_rates))
    train_pred = pred_miss_rates[:split]
    train_true = true_miss_rates[:split]
    val_pred = pred_miss_rates[split:]
    val_true = true_miss_rates[split:]

    results = []
    for i, (lam, alpha, lr, n_ep) in enumerate(combos):
        _print_progress(i, total, "DQN")
        try:
            ctrl = DQNController(lambda_tradeoff=lam, alpha_switch=alpha, lr=lr)
            ctrl.train(train_pred, train_true, t1_eval, t2_eval, n_episodes=n_ep)
            m = evaluate(ctrl, val_pred, val_true, MODELS,
                         t1_eval, t2_eval, epistemic_unc)
            m["config"] = {"lambda_tradeoff": lam, "alpha_switch": alpha,
                           "lr": lr, "n_train_episodes": n_ep}
            results.append(m)
        except Exception as e:
            print(f"  DQN failed (λ={lam}, α={alpha}, lr={lr}): {e}")
            continue

    if not results:
        print("  [DQN] No successful runs.")
        return None, None, []

    ranked = _filter_and_rank(results, min_adequate)
    best = ranked[0] if ranked else results[0]
    print(f"  Best: λ={best['config']['lambda_tradeoff']}, "
          f"α={best['config']['alpha_switch']}, lr={best['config']['lr']} | "
          f"energy={best['avg_energy_mj']:.1f}mJ | adequate={best['adequate_rate']:.3f}")
    return best["config"], best, ranked


# ═══════════════════════════════════════════════════════════════════════════════
# Proxy search (uncertainty-based, only if epistemic_unc available)
# ═══════════════════════════════════════════════════════════════════════════════

def search_proxy(pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
                 epistemic_unc: np.ndarray,
                 t1_eval: float, t2_eval: float,
                 min_adequate: float = 0.85) -> tuple:
    """Grid search over proxy (uncertainty-based) controller thresholds.

    ~190 combinations. Only callable when epistemic_unc is available.

    Returns:
        (best_config, best_metrics, all_results_sorted)
    """
    low_vals = np.linspace(0.01, 0.30, 20)
    high_vals = np.linspace(0.05, 0.50, 20)

    combos = [(lo, hi) for lo in low_vals for hi in high_vals if hi > lo]
    total = len(combos)
    print(f"[Proxy Search] {total} combinations")

    results = []
    for i, (lo, hi) in enumerate(combos):
        _print_progress(i, total, "Proxy")
        ctrl = ProxyController(unc_threshold_low=lo, unc_threshold_high=hi)
        ctrl.set_uncertainty(epistemic_unc)
        m = evaluate(ctrl, pred_miss_rates, true_miss_rates, MODELS,
                     t1_eval, t2_eval, epistemic_unc)
        m["config"] = {"unc_threshold_low": lo, "unc_threshold_high": hi}
        results.append(m)

    ranked = _filter_and_rank(results, min_adequate)
    best = ranked[0] if ranked else results[0]
    print(f"  Best: lo={best['config']['unc_threshold_low']:.3f}, "
          f"hi={best['config']['unc_threshold_high']:.3f} | "
          f"energy={best['avg_energy_mj']:.1f}mJ | adequate={best['adequate_rate']:.3f}")
    return best["config"], best, ranked


# ═══════════════════════════════════════════════════════════════════════════════
# Pareto frontier sweep
# ═══════════════════════════════════════════════════════════════════════════════

def pareto_sweep(pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
                 t1_eval: float, t2_eval: float,
                 models_config: dict = None,
                 t1_range: np.ndarray = None,
                 t2_range: np.ndarray = None,
                 epistemic_unc: np.ndarray = None) -> list:
    """Sweep the adequate_rate constraint from 0.70 to 0.95 and find the
    best threshold controller (minimum energy) at each constraint level.

    Args:
        pred_miss_rates: Predicted miss rates array.
        true_miss_rates: True miss rates array.
        t1_eval, t2_eval: Fixed evaluation thresholds for adequacy scoring.
        models_config: MODELS dict (defaults to controller.models.MODELS).
        t1_range: Array of t1 values to search (default: linspace(0.01, 0.50, 30)).
        t2_range: Array of t2 values to search (default: linspace(0.05, 0.80, 30)).
        epistemic_unc: Optional epistemic uncertainty array.

    Returns:
        List of (constraint, best_energy_savings_pct, best_adequate_rate, best_config)
        tuples — one per constraint level.
    """
    if models_config is None:
        models_config = MODELS
    if t1_range is None:
        t1_range = np.linspace(0.01, 0.50, 30)
    if t2_range is None:
        t2_range = np.linspace(0.05, 0.80, 30)

    constraints = np.arange(0.70, 0.96, 0.05)

    # Run threshold grid once and cache all results
    combos = [(t1, t2) for t1 in t1_range for t2 in t2_range if t2 > t1]
    print(f"[Pareto Sweep] evaluating {len(combos)} threshold combos "
          f"across {len(constraints)} constraint levels")

    all_evals = []
    for t1, t2 in combos:
        ctrl = ThresholdController(t1=t1, t2=t2)
        m = evaluate(ctrl, pred_miss_rates, true_miss_rates, models_config,
                     t1_eval, t2_eval, epistemic_unc)
        m["config"] = {"t1": t1, "t2": t2}
        all_evals.append(m)

    # For each constraint level, find the best feasible config
    frontier = []
    for constraint in constraints:
        feasible = [r for r in all_evals if r["adequate_rate"] >= constraint]
        if feasible:
            best = min(feasible, key=lambda r: r["avg_energy_mj"])
            frontier.append((
                float(constraint),
                float(best["energy_savings_pct"]),
                float(best["adequate_rate"]),
                best["config"],
            ))
        else:
            # No config meets this constraint — report the closest
            closest = max(all_evals, key=lambda r: r["adequate_rate"])
            frontier.append((
                float(constraint),
                float(closest["energy_savings_pct"]),
                float(closest["adequate_rate"]),
                closest["config"],
            ))

    print(f"  Pareto frontier: {len(frontier)} points")
    return frontier
