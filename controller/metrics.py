"""
Evaluation metrics for controller simulation.

Runs a controller over a prediction stream, computes energy consumption,
detection adequacy, switching frequency, and comparison against the oracle.
"""

import numpy as np

from .models import (MODELS, FRAMES_PER_WINDOW, required_model,
                     energy_per_window, switching_energy, step_energy)
from .controllers import OracleController


def evaluate(ctrl, pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
             models_config: dict, t1: float, t2: float,
             epistemic_unc: np.ndarray = None) -> dict:
    """Run a controller over the prediction stream and compute all metrics.

    Args:
        ctrl: Controller instance (must implement select_model, reset, name).
        pred_miss_rates: Array of LSTM-predicted miss rates.
        true_miss_rates: Array of ground-truth miss rates.
        models_config: Model hardware specs (the MODELS dict from models.py).
        t1: Lower threshold for required_model (easy/moderate boundary).
        t2: Upper threshold for required_model (moderate/hard boundary).
        epistemic_unc: Optional epistemic uncertainty array (for ProxyController).

    Returns:
        Dict of metrics including energy, adequacy, switching, and model distribution.
    """
    n = len(pred_miss_rates)
    assert len(true_miss_rates) == n, "Prediction and truth arrays must match"

    ctrl.reset()

    # If it's an oracle, inject ground truth
    if isinstance(ctrl, OracleController):
        ctrl.set_ground_truth(true_miss_rates)
        ctrl.t1 = t1
        ctrl.t2 = t2

    # If it's a proxy controller, inject uncertainty
    from .controllers import ProxyController
    if isinstance(ctrl, ProxyController) and epistemic_unc is not None:
        ctrl.set_uncertainty(epistemic_unc)

    # ── Run controller ────────────────────────────────────────────────────
    selections = np.zeros(n, dtype=int)
    energies = np.zeros(n)
    state = {}

    prev_model = 1  # start with small
    for t in range(n):
        model = ctrl.select_model(pred_miss_rates[t], t, state)
        selections[t] = model
        energies[t] = step_energy(model, prev_model)
        prev_model = model

    # ── Oracle reference ──────────────────────────────────────────────────
    oracle = OracleController(t1=t1, t2=t2)
    oracle.set_ground_truth(true_miss_rates)
    oracle_selections = np.array([
        required_model(true_miss_rates[t], t1, t2) for t in range(n)
    ])
    oracle_energies = np.zeros(n)
    prev_oracle = 1
    for t in range(n):
        oracle_energies[t] = step_energy(oracle_selections[t], prev_oracle)
        prev_oracle = oracle_selections[t]

    # ── Always-medium reference ───────────────────────────────────────────
    medium_energy = energy_per_window(2)  # no switching for constant model

    # ── Compute metrics ───────────────────────────────────────────────────
    avg_energy = energies.mean()
    oracle_avg = oracle_energies.mean()

    # Required models for each frame based on true miss rates
    required_models = np.array([required_model(true_miss_rates[t], t1, t2)
                                for t in range(n)])

    adequate = (selections >= required_models).astype(float)
    under = (selections < required_models).astype(float)
    over = (selections > required_models).astype(float)
    correct = (selections == required_models).astype(float)

    # Switching count
    switches = np.sum(selections[1:] != selections[:-1])

    # Model distribution
    model_dist = {}
    for m in range(3):
        model_dist[models_config[m]["name"]] = np.mean(selections == m) * 100.0

    # Average latency
    avg_latency = np.mean([models_config[selections[t]]["latency_ms"]
                           for t in range(n)])

    # Oracle gap: 0 = matches oracle, 1 = no better than always-medium
    denom = medium_energy - oracle_avg
    oracle_gap = (avg_energy - oracle_avg) / denom if abs(denom) > 1e-9 else 0.0

    return {
        "controller": ctrl.name(),
        "avg_energy_mj": avg_energy,
        "energy_savings_pct": 100.0 * (1.0 - avg_energy / medium_energy),
        "oracle_gap": oracle_gap,
        "switches_per_100": switches / n * 100.0,
        "adequate_rate": adequate.mean(),
        "under_provision_rate": under.mean(),
        "over_provision_rate": over.mean(),
        "correct_provision_rate": correct.mean(),
        "model_distribution": model_dist,
        "avg_latency_ms": avg_latency,
        "n_frames": n,
        "selections": selections,
        "energies": energies,
        "oracle_selections": oracle_selections,
    }
