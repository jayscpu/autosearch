"""
Hardware constants for YOLO11 model tiers on Jetson Orin Nano (15W TensorRT FP16).

All measurements taken with clocks locked. These values are used throughout
the controller simulation to compute energy costs and switching overhead.
"""

import numpy as np

# ── Jetson Orin Nano 15W benchmarks (TensorRT FP16, clocks locked) ───────────

MODELS = {
    0: {  # nano@640
        "name": "nano",
        "latency_ms": 12.89,
        "latency_std_ms": 0.13,
        "fps": 77.6,
        "power_mw": 6622,
        "energy_mj_per_frame": 85.36,
    },
    1: {  # small@640
        "name": "small",
        "latency_ms": 16.58,
        "latency_std_ms": 0.12,
        "fps": 60.3,
        "power_mw": 7760,
        "energy_mj_per_frame": 128.65,
    },
    2: {  # medium@640
        "name": "medium",
        "latency_ms": 26.40,
        "latency_std_ms": 0.16,
        "fps": 37.9,
        "power_mw": 9410,
        "energy_mj_per_frame": 248.46,
    },
}

IDLE_POWER_MW = 4013

# Switching costs: (from_tier, to_tier) -> (total_latency_ms, overhead_ms)
# Upward switches incur 3-5ms extra overhead; downward switches are free.
SWITCH_COSTS = {
    (0, 1): {"total_ms": 20.1, "overhead_ms": 3.5},
    (0, 2): {"total_ms": 31.6, "overhead_ms": 5.3},
    (1, 0): {"total_ms": 12.9, "overhead_ms": 0.0},
    (1, 2): {"total_ms": 26.3, "overhead_ms": 0.0},
    (2, 0): {"total_ms": 12.9, "overhead_ms": 0.0},
    (2, 1): {"total_ms": 16.6, "overhead_ms": 0.0},
}

# Camera runs at 30fps. LSTM predictor evaluates at 2fps (every 15th frame).
# Each row in prediction CSVs = one 0.5s decision window (~15 detector frames).
FRAMES_PER_WINDOW = 15
DETECTOR_FPS = 30
PREDICTOR_FPS = 2


def energy_per_window(model_idx: int) -> float:
    """Energy (mJ) for one decision window (15 detector frames) at given model tier."""
    return MODELS[model_idx]["energy_mj_per_frame"] * FRAMES_PER_WINDOW


def switching_energy(from_idx: int, to_idx: int) -> float:
    """Energy cost (mJ) of switching between model tiers.

    Computed as: switch_overhead_ms * average_power_mW / 1000.
    Returns 0 if no switch occurs.
    """
    if from_idx == to_idx:
        return 0.0
    cost = SWITCH_COSTS[(from_idx, to_idx)]
    avg_power = (MODELS[from_idx]["power_mw"] + MODELS[to_idx]["power_mw"]) / 2.0
    return cost["overhead_ms"] * avg_power / 1000.0


def step_energy(model_idx: int, prev_model_idx: int) -> float:
    """Total energy (mJ) for one simulation step: window energy + switching cost."""
    return energy_per_window(model_idx) + switching_energy(prev_model_idx, model_idx)


def required_model(true_miss_rate: float, t1: float, t2: float) -> int:
    """Oracle logic: return the minimum adequate model tier for a given true miss rate.

    - 0 (nano)   if true_miss_rate < t1  (easy scene)
    - 1 (small)  if true_miss_rate < t2  (moderate scene)
    - 2 (medium) otherwise               (hard scene)
    """
    if true_miss_rate < t1:
        return 0
    elif true_miss_rate < t2:
        return 1
    else:
        return 2
