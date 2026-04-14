# Task: Multi-Step MPC Controller with Per-Step LSTM Predictions

Create a folder `refinedsys/` in the autosearch repo root. This task has two parts: (1) modify prediction output to preserve per-step predictions, and (2) build a real multi-step MPC controller that consumes them.

## Context

The current system has an LSTM that produces 5 sub-window predictions autoregressively (each covering 3 seconds, totaling 15 seconds of forecast horizon). However, these 5 predictions are **averaged into a single value** before reaching the controller. This wastes the temporal granularity the LSTM already provides.

The averaging happens in one line in `new_features/feature_selection.py`:
```python
def predict(model, X, device):
    ...
    return np.concatenate(out, axis=0).mean(axis=1)  # <-- THIS LINE averages 5 steps into 1
```

The model output shape is `(N_windows, 5, 3)` — 5 steps × 3 targets (miss_rate_nano, recovery_ns, recovery_sm). After `.mean(axis=1)` it becomes `(N_windows, 3)`.

Similarly, ground truth is averaged: `y_w_mean = y_w_raw.mean(axis=1)` in the `prepare()` function.

The current controllers (DirectThreshold, BayesMPC, RichMPC) are ALL stateless — they make each decision independently with no memory of previous selections and no switching costs.

## Part 1: Save Per-Step Predictions

Create `refinedsys/generate_multistep_predictions.py` that:

1. Loads the trained k=80 model (reuse the exact training code from `new_features/feature_selection.py` — same `MultiOutputLSTM` class, same `CFG` dict, same `prepare()` function, same 3 seeds)
2. Instead of averaging predictions, saves ALL 5 steps per window

### Key parameters (do NOT change these):
```python
CFG = {
    "window": 30,          # 30 frames context (15 seconds at 2fps)
    "horizon": 30,         # 30 frames forecast horizon (15 seconds)
    "sub_window": 6,       # 6 frames per sub-window (3 seconds)
    # → n_steps = horizon / sub_window = 5
    "hidden_size": 128,
    "n_layers": 2,
    "dropout": 0.4,
    "train_intersections": [
        "Bellevue_150th_Eastgate", "Bellevue_150th_Newport",
        "Bellevue_Bellevue_NE8th", "Bellevue_116th_NE12th",
    ],
    "test_intersection": "Bellevue_150th_SE38th",
    "train_fraction": 0.50, "earlystop_fraction": 0.60,
    "warmup_frames": 400,
}
```

### Output format
Save per-seed CSV files: `refinedsys/predictions_multistep_seed{42,43,44}.csv`

Columns:
```
frame_idx, split, intersection,
pred_nano_s0, pred_nano_s1, pred_nano_s2, pred_nano_s3, pred_nano_s4,
pred_recovery_ns_s0, ..., pred_recovery_ns_s4,
pred_recovery_sm_s0, ..., pred_recovery_sm_s4,
true_nano_s0, ..., true_nano_s4,
true_recovery_ns_s0, ..., true_recovery_ns_s4,
true_recovery_sm_s0, ..., true_recovery_sm_s4
```

Each `_s0` through `_s4` is one decoder step (covering the next 3 seconds each, 15 seconds total).

### Important details:
- The model is trained on **normalized** targets. Predictions must be **denormalized** before saving: `pred_real = pred_normalized * t_std + t_mean`. Apply this per-step, not after averaging.
- Ground truth per-step values are in `y_w_raw` (shape N×5×3) and `y_c_raw` (shape N×5×3) — these are already available in the `prepare()` function's return dict, just not currently exposed individually.
- Also save the averaged versions for backwards compatibility verification: `pred_nano_avg, pred_recovery_ns_avg, pred_recovery_sm_avg` (mean over s0-s4). These should match the existing `predictions_best_seed*.csv` values to within floating point tolerance.
- Features CSV: `new_features/pod_features_merged_95.csv`
- Feature list for k=80: use the top 80 from `new_features/perm_importance_95.tsv` (ranks 1-80)

### Feature list (k=80):
To get the k=80 feature list, read `new_features/perm_importance_95.tsv`, sort by rank, take ranks 1-80:
```python
imp = pd.read_csv("new_features/perm_importance_95.tsv", sep="\t")
feat_cols = imp.sort_values("rank").head(80)["feature"].tolist()
```

## Part 2: Multi-Step MPC Controller

Create `refinedsys/run_multistep_controller.py` that implements a **real** multi-step MPC.

### How the current (broken) BayesMPC works:
```python
# Current: uses SAME prediction for all H steps. Stateless. No switching cost.
for seq in product(range(3), repeat=H):
    cost = 0
    for step in range(H):
        m = seq[step]
        cost += e_norm[m]
        cost += lu * max(0, pm[m] - T)  # pm is the SAME for every step
    if cost < best_cost:
        best_first = seq[0]
```

### How the new multi-step MPC should work:

```python
# New: uses DIFFERENT predictions per step. Tracks current model. Adds switching cost.
def select_model(self, per_step_preds, current_model, T):
    """
    per_step_preds: dict of {step_idx: {0: pred_nano, 1: pred_small, 2: pred_medium}}
        - 5 steps, each with per-tier miss rate predictions
    current_model: int (0/1/2) — what model is currently running
    T: adequacy threshold
    """
    H = 5  # use all 5 decoder steps
    seqs = list(product(range(3), repeat=H))

    best_cost = float("inf")
    best_first = current_model

    for seq in seqs:  # 3^5 = 243 sequences
        cost = 0.0
        prev = current_model  # track state!
        for step in range(H):
            m = seq[step]
            # Energy cost (normalized)
            cost += e_norm[m]
            # Under-provisioning penalty using THIS STEP's prediction
            cost += lu * max(0, per_step_preds[step][m] - T)
            # Switching cost — penalize changing model
            if m != prev:
                cost += w_switch
            prev = m
        if cost < best_cost:
            best_cost = cost
            best_first = seq[0]  # execute only first step

    return best_first
```

### Key differences from current system:
1. **Per-step predictions**: step 0 might predict easy, step 3 might predict hard — the MPC can plan for this transition
2. **Switching cost** (`w_switch`): penalizes changing models between steps. Without this, multi-step planning offers no benefit over greedy.
3. **State tracking**: `current_model` carries across windows — the MPC knows what it selected last time

### Computing per-tier miss rates from recovery predictions:
The LSTM predicts (miss_rate_nano, recovery_ns, recovery_sm). Convert to per-tier:
```python
pred_nano = pred_nano_s{step}
pred_small = pred_nano_s{step} - pred_recovery_ns_s{step}
pred_medium = pred_small - pred_recovery_sm_s{step}
```

### Controllers to implement:

1. **MultiStepMPC**: As described above. Hyperparameters to grid-search on train split:
   - `lu` (under-provisioning weight): [1.0, 3.0, 5.0, 10.0, 20.0]
   - `w_switch` (switching penalty): [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
   - H is fixed at 5 (all decoder steps)

2. **MultiStepMPC_RichVariant**: Same but adds over-provisioning penalty:
   - `cost += lo * max(0, T - per_step_preds[step][m])` for using a bigger model than needed
   - Grid-search `lo`: [0.1, 0.3, 0.5, 1.0]

3. Keep the existing stateless controllers as baselines for comparison:
   - DirectThreshold (from existing code)
   - BayesMPC H=1 stateless (from existing code) — rename to "GreedyBayes" for clarity
   - Oracle, AlwaysNano, AlwaysSmall, AlwaysMedium

### Evaluation:

Use the EXACT same evaluation protocol as `new_features/run_controller_k80.py`:
- 60/40 train/test split per intersection (chronological)
- Hyperparameters optimized on train split only
- Report on test split
- Same adequacy metric (adequate = selection ≥ oracle recommendation)
- Same thresholds: T ∈ {0.30, 0.35, 0.40, 0.50}
- Energy constants:
  ```python
  E_NANO = 15 * 85.36      # mJ per window (15 frames)
  E_SMALL = 15 * 128.65
  E_MEDIUM = 15 * 248.46
  ```

### Additional metric to add:
- `switches_per_100`: already computed in existing code, keep it
- The multi-step MPC should show **fewer switches** than stateless controllers at similar adequacy

### Output:
Save results to `refinedsys/controller_multistep_results.tsv` with the same columns as the existing `controller_k80_results.tsv`:
```
threshold, n_solvable, n_total, controller, energy_savings_pct, adequate_rate,
correct_rate, over_provision_rate, mean_miss_rate, pct_nano, pct_small,
pct_medium, switches_per_100
```

Also generate a comparison plot `refinedsys/comparison_plot.png` showing the adequacy-savings tradeoff at T=0.50, including both old (stateless) and new (multi-step) controllers.

## File structure:
```
refinedsys/
├── generate_multistep_predictions.py    # Part 1
├── run_multistep_controller.py          # Part 2
├── predictions_multistep_seed42.csv     # Generated by Part 1
├── predictions_multistep_seed43.csv
├── predictions_multistep_seed44.csv
├── controller_multistep_results.tsv     # Generated by Part 2
└── comparison_plot.png                  # Generated by Part 2
```

## Dependencies:
All already installed in the repo: numpy, pandas, torch, scikit-learn, matplotlib

## Critical: Do NOT change:
- The LSTM architecture (MultiOutputLSTM class)
- The training protocol (seeds, splits, hyperparameters)
- The feature set (k=80)
- The energy constants
- The adequacy metric definition
- The evaluation protocol

The ONLY changes are:
1. Stop averaging the 5 decoder steps before saving predictions
2. Build a controller that actually uses the 5 per-step predictions with switching costs and state tracking
