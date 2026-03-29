"""
Budget-constrained controller simulation.

From thesis Chapter 4, Section 4.2.3: runs a controller under a fixed total
energy budget. When the budget is exhausted, the system enters degraded mode
(forced nano) for the remainder of the sequence.
"""

import numpy as np

from .models import (MODELS, FRAMES_PER_WINDOW, required_model,
                     energy_per_window, step_energy)


def run_budget_constrained(ctrl, pred_stream: np.ndarray, true_stream: np.ndarray,
                           models_config: dict, t1_eval: float, t2_eval: float,
                           total_budget_j: float, kappa: float = 2.0,
                           epistemic_unc: np.ndarray = None) -> dict:
    """Run a controller under a total energy budget constraint.

    When cumulative energy consumption reaches the budget, the system enters
    degraded mode: forced to nano for all remaining frames.

    Args:
        ctrl: Controller instance.
        pred_stream: Predicted miss rates array.
        true_stream: True miss rates array.
        models_config: MODELS dict from models.py.
        t1_eval, t2_eval: Fixed evaluation thresholds for adequacy scoring.
        total_budget_j: Total energy budget in joules (e.g., n * small_energy / 1000).
        kappa: Budget slack sensitivity — controls how aggressively the controller
            downgrades when remaining budget is tight.
        epistemic_unc: Optional epistemic uncertainty array.

    Returns:
        Dict with standard metrics plus budget-specific metrics.
    """
    n = len(pred_stream)
    total_budget_mj = total_budget_j * 1000.0  # convert to millijoules

    ctrl.reset()

    # Handle oracle and proxy controllers
    from .controllers import OracleController, ProxyController
    if isinstance(ctrl, OracleController):
        ctrl.set_ground_truth(true_stream)
        ctrl.t1 = t1_eval
        ctrl.t2 = t2_eval
    if isinstance(ctrl, ProxyController) and epistemic_unc is not None:
        ctrl.set_uncertainty(epistemic_unc)

    selections = np.zeros(n, dtype=int)
    energies = np.zeros(n)
    e_consumed = 0.0
    degraded = np.zeros(n, dtype=bool)
    budget_exhausted_at = -1
    prev_model = 1
    state = {}

    for t in range(n):
        # Check if budget is exhausted
        if e_consumed >= total_budget_mj:
            # Degraded mode: force nano
            model = 0
            if budget_exhausted_at < 0:
                budget_exhausted_at = t
            degraded[t] = True
        else:
            # Normal operation
            model = ctrl.select_model(pred_stream[t], t, state)

            # Budget slack check: if remaining budget is tight, prefer cheaper model
            remaining = total_budget_mj - e_consumed
            remaining_frames = n - t
            if remaining_frames > 0:
                avg_budget_per_frame = remaining / remaining_frames
                # If selected model's energy exceeds what the budget can sustain,
                # downgrade to maintain budget
                if energy_per_window(model) > avg_budget_per_frame * kappa:
                    # Try cheaper models
                    for cheaper in range(model - 1, -1, -1):
                        if energy_per_window(cheaper) <= avg_budget_per_frame * kappa:
                            model = cheaper
                            break
                    else:
                        model = 0  # force nano if nothing fits

        selections[t] = model
        e_step = step_energy(model, prev_model)
        energies[t] = e_step
        e_consumed += e_step
        prev_model = model

    # ── Compute standard metrics (using eval thresholds for adequacy) ─────
    avg_energy = energies.mean()
    medium_energy = energy_per_window(2)

    required_models = np.array([required_model(true_stream[t], t1_eval, t2_eval)
                                for t in range(n)])

    adequate = (selections >= required_models).astype(float)
    under = (selections < required_models).astype(float)
    over = (selections > required_models).astype(float)
    correct = (selections == required_models).astype(float)

    switches = np.sum(selections[1:] != selections[:-1])

    model_dist = {}
    for m in range(3):
        model_dist[models_config[m]["name"]] = np.mean(selections == m) * 100.0

    avg_latency = np.mean([models_config[selections[t]]["latency_ms"]
                           for t in range(n)])

    return {
        "controller": ctrl.name() + f" [budget k={kappa}]",
        "avg_energy_mj": avg_energy,
        "energy_savings_pct": 100.0 * (1.0 - avg_energy / medium_energy),
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
        # Budget-specific metrics
        "time_in_degraded_pct": degraded.mean() * 100.0,
        "final_budget_utilization": e_consumed / total_budget_mj,
        "budget_exhausted_at_frame": budget_exhausted_at,
        "total_budget_j": total_budget_j,
        "total_consumed_j": e_consumed / 1000.0,
        "kappa": kappa,
    }
