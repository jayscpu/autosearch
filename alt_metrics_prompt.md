# Task: Alternative Evaluation Metrics for Controllers

Create `refinedsys_metrics/run_alternative_metrics.py` in the autosearch repo.

## Purpose

The current adequacy metric (binary: did you pick ≥ oracle?) rewards conservative over-provisioning. DirectThreshold wins by copying the oracle's logic with safety margins. This experiment tests whether energy-aware controllers perform better under metrics that penalize wasteful over-provisioning.

## Data

Load predictions from `refinedsys/predictions_multistep_seed{42,43,44}.csv` (already generated). Average across 3 seeds. Derive per-tier miss rates from recovery predictions (same as existing controllers). Use averaged predictions (pred_nano_avg, etc.) — no per-step needed here.

## Controllers to evaluate

Reuse the exact same controller logic and hyperparameters from `refinedsys/run_multistep_controller.py`:

1. **AlwaysNano** — always model 0
2. **AlwaysSmall** — always model 1
3. **AlwaysMedium** — always model 2
4. **Oracle** — cheapest model with true miss rate < T
5. **DirectThreshold** — threshold on predicted miss rates with margins (mn, ms)
6. **EnergyPenalty** — picks model minimizing `e_norm[m] + lu * max(0, pred[m] - T)` (the renamed BayesMPC/GreedyBayes)

Same train/test split (60/40 chronological per intersection).
Same margin grid: [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
Same lu grid: [1.0, 3.0, 5.0, 10.0, 20.0]
Same energy constants: E_NANO = 15*85.36, E_SMALL = 15*128.65, E_MEDIUM = 15*248.46
Same thresholds: T ∈ {0.30, 0.35, 0.40, 0.50}

**Important**: Hyperparameters must be re-optimized per metric. For each metric below, grid-search on the train split to maximize that specific metric, then report on the test split.

## Metrics to compute

For every controller at every threshold, compute ALL of the following. The difference is which metric is used for hyperparameter optimization.

### Metric 1: Binary Adequacy (existing — baseline)
```python
adequate = (selection >= oracle_selection).mean()
savings = 1 - mean(energy[selection]) / E_MEDIUM
```
Optimize: `score = adequate_rate/100 + 0.5 * savings/100`

### Metric 2: Efficiency-Weighted Quality (EWQ)
Continuous score per window that rewards both low miss rate AND low energy:
```python
# Per window i:
actual_mr = true_miss_rate for the selected model
energy_ratio = ENERGY[selection] / E_MEDIUM  # 1.0 for medium, ~0.34 for nano
quality = max(0, 1 - actual_mr / T)  # 1.0 = perfect, 0.0 = at threshold, capped at 0
efficiency = 1 - energy_ratio  # 1.0 = free, 0.0 = medium cost

ewq_per_window = quality * (1 + efficiency)  
# Range: [0, 2]. Best: perfect quality with zero energy.
# Medium on easy scene: quality=1.0, efficiency=0.0 → ewq=1.0
# Nano on easy scene: quality=1.0, efficiency=0.66 → ewq=1.66 (better!)
# Nano on hard scene: quality=0.0, efficiency=0.66 → ewq=0.0 (punished)

ewq = mean(ewq_per_window)  # over solvable windows only
```
Optimize: `score = ewq`

### Metric 3: Miss-Rate-Aware Savings (MRAS)
Energy savings conditional on acceptable detection:
```python
# Per window i:
actual_mr = true_miss_rate for the selected model
acceptable = actual_mr < T  # boolean
energy_saved = (E_MEDIUM - ENERGY[selection]) / E_MEDIUM

# Only count savings when detection is acceptable
mras_per_window = energy_saved if acceptable else -penalty
# penalty = 1.0 (symmetric: bad detection costs as much as medium-tier energy)

mras = mean(mras_per_window)  # over solvable windows
```
Optimize: `score = mras`

### Metric 4: Pareto Score
Combined score that treats adequacy and savings as equally important:
```python
pareto = adequate_rate/100 * energy_savings_pct/100
# Range: [0, 1]. Maximized when both are high.
# DT at T=0.50: 0.87 * 0.33 = 0.29
# EP at T=0.50: 0.81 * 0.43 = 0.35 (better!)
# Oracle: 1.00 * 0.42 = 0.42
```
Optimize: `score = pareto`

### Metric 5: Threshold-Relative Miss Rate (TRMR)
How close to the threshold did you actually get? Rewards controllers that use their budget wisely — getting close to T without exceeding it:
```python
# Per window i:
actual_mr = true_miss_rate for the selected model
headroom = T - actual_mr  # positive = under threshold, negative = over
energy_ratio = ENERGY[selection] / E_MEDIUM

# Reward: being just under threshold with cheap model
# Punish: being way under threshold with expensive model (wasted capacity)
# Punish: exceeding threshold (missed detections)
if headroom >= 0:
    trmr_per_window = (1 - energy_ratio) * (1 - headroom/T)
    # Low energy + close to T → high score
    # Low energy + far below T → moderate (detection was easy, savings good)
    # High energy + far below T → low (wasted capacity)
else:
    trmr_per_window = headroom  # negative penalty proportional to violation

trmr = mean(trmr_per_window)  # over solvable windows
```
Optimize: `score = trmr`

## Output

### Table 1: controller_alt_metrics_results.tsv
One row per (threshold, controller, optimization_metric) combination.
Columns:
```
threshold, controller, opt_metric, 
energy_savings_pct, adequate_rate, ewq, mras, pareto, trmr,
pct_nano, pct_small, pct_medium, switches_per_100
```

Note: Every row computes ALL metrics, but the `opt_metric` column says which metric was used to select hyperparameters for that row.

### Table 2: Summary printed to stdout
For each threshold, show which controller wins under each metric:
```
T=0.50:
  Metric              | Winner          | Score  | DT score | EP score
  Binary Adequacy      | DirectThresh    | 0.87   | 0.87     | 0.81
  EWQ                  | EnergyPenalty   | 1.42   | 1.31     | 1.42
  ...
```

### Plot: `refinedsys_metrics/metric_comparison_plot.png`
2×2 subplot grid (one per threshold). Each subplot shows controllers as points on adequacy (x) vs savings (y), with point color indicating which metric they were optimized for. 

## File structure
```
refinedsys_metrics/
├── run_alternative_metrics.py
├── controller_alt_metrics_results.tsv    (generated)
└── metric_comparison_plot.png            (generated)
```

## Critical
- Re-optimize hyperparameters PER METRIC on train split
- Report ALL metrics for every controller regardless of optimization metric
- Exclude unsolvable windows (same as existing)
- Same oracle, same energy constants, same train/test split
