# ECHO Bellevue Experiment — Autonomous Search (Regression)

## Context

Multi-intersection generalization experiment for ECHO. Uses 5 Bellevue traffic
camera intersections (~101 hours total). Tests whether spatial features trained
on 4 intersections generalize to unseen intersection 3.

Two-sweep design: each run trains a single model type.
- **Sweep A (LSTM):** PlainLSTM only (3-seed ensemble, MSE loss)
- **Sweep B (Evidential):** EvidentialLSTM only (NIG loss, single seed)

Data is pre-extracted by `pod_pipeline.py`. Work only with the CSVs.

## Repo Structure

```
autosearch/second_autosearch/
├── docs/
│   └── pod_program.md              ← YOU ARE HERE
├── src/
│   ├── pod_train.py                ← Sweep A (LSTM) — THE FILE YOU EDIT
│   ├── pod_train_evid.py           ← Sweep B (Evidential) — copy, diverges during sweep
│   ├── pod_features.py             ← shared feature list
│   ├── pod_pipeline.py             ← extraction pipeline (already run)
│   ├── pod_download.sh             ← data download script
│   └── pod_setup.sh                ← environment setup script
├── data/
│   ├── pod_features_all.csv        ← combined features (DO NOT MODIFY)
│   ├── pod_features_{intersection}.csv  ← per-intersection features
│   └── pod_detections_{intersection}.csv
└── results/
    ├── pod_results_lstm.tsv        ← Sweep A experiment log
    └── pod_results_evid.tsv        ← Sweep B experiment log
```

## Experimental Design

### Two-Sweep Design

**Sweep A (LSTM):** `python src/pod_train.py --mode lstm --results-file results/pod_results_lstm.tsv`
- Trains only PlainLSTM (3-seed ensemble, MSE loss)
- Keep/discard based on LSTM mse_within with cls_trans > 0.50 filter
- Optimizes: features, window, horizon, architecture, LR, dropout, batch_size, thresholds

**Sweep B (Evidential):** `python src/pod_train.py --mode evidential --results-file results/pod_results_evid.tsv`
- Trains only EvidentialLSTM (NIG loss, single seed)
- Keep/discard based on EvidentialLSTM mse_within with cls_trans > 0.50 AND unc_sep > 0 filter
- Optimizes: same shared hyperparams PLUS lambda1

**Both sweeps run in parallel** in two separate terminal windows. Each sweep edits its own copy of pod_train.py to avoid git conflicts:
- Sweep A: works on second_autosearch/src/pod_train.py, logs to second_autosearch/results/pod_results_lstm.tsv
- Sweep B: works on second_autosearch/src/pod_train_evid.py (copy), logs to second_autosearch/results/pod_results_evid.tsv

**After both sweeps:** Run RF and LSTM+RF once at each optimal config for the thesis comparison table.

### Data Splits

**Training set:** First 50% of intersections 1, 2, 4, 5 (by temporal order)
**Early-stop set:** Next 10% (50–60%) of intersections 1, 2, 4, 5 (checkpoint selection only)
**Within-camera val:** Last 40% (60–100%) of intersections 1, 2, 4, 5 (never seen during training)
**Cross-camera val:** All of intersection 3 (Bellevue_150th_SE38th)
**Combined val:** Within + Cross combined

The 3-way split (50% train / 10% early-stop / 40% within-eval) ensures that
`mse_within` is an honest metric — no model sees within-val during training or
checkpoint selection.

**Multi-step prediction:** The horizon is divided into sub-windows. Each step
predicts mean miss_rate over one sub-window. Default: horizon=30, sub_window=6,
n_steps=5.

## Primary Metrics

**Regression (both modes):**
1. `mse_within` / `mse_cross` — mean squared error on continuous miss_rate
2. `mae_within` / `mae_cross` — mean absolute error
3. `cls_acc_within` / `cls_acc_cross` — 3-class accuracy (easy/moderate/hard)
4. `cls_trans_within` / `cls_trans_cross` — 3-class transition accuracy

**Evidential mode additionally:**
5. `unc_sep_within` / `unc_sep_cross` — uncertainty separation (mean epistemic
   uncertainty of incorrect minus correct predictions). MUST be > 0.

**Ranking (same for both modes):**
- Filter: cls_trans > 0.50 on all val sets
- Primary: minimize mse_within
- Tiebreaker: maximize cls_acc
- Evidential additionally: require unc_sep > 0 (uncertainty must be calibrated)

## Available Intersections

| # | Intersection | Role |
|---|---|---|
| 1 | Bellevue_150th_Eastgate | Training (50% train / 10% early-stop / 40% within-val) |
| 2 | Bellevue_150th_Newport | Training (50% train / 10% early-stop / 40% within-val) |
| 3 | Bellevue_150th_SE38th | Cross-camera test (100% held out) |
| 4 | Bellevue_Bellevue_NE8th | Training (50% train / 10% early-stop / 40% within-val) |
| 5 | Bellevue_116th_NE12th | Training (50% train / 10% early-stop / 40% within-val) |

Intersection 3 is the cross-camera held-out test — same 150th corridor as
intersections 1+2, making it a realistic "new camera, same road" generalization
test.

## Search Space

### Models
- Each run trains ONE model based on `--mode` flag (lstm or evidential)
- LSTM mode: PlainLSTM 3-seed ensemble, keep/discard on mse_within
- Evidential mode: EvidentialLSTM single seed, keep/discard on mse_within + unc_sep > 0
- Shared hyperparams affect both; λ₁ only affects evidential mode

### Features
- Start with top-35 Spearman features from classification search (proven best)
- Also try all 65, all 75, top-20, top-25, top-30, top-40
- The optimal subset for regression may differ from classification
- **Feature selection methods** (set flags in CONFIG, rankings print to stderr):
  - `spearman_feature_selection`: absolute Spearman rank correlation with miss_rate
  - `mi_feature_selection`: mutual information regression on RF summary features
  - `lasso_feature_selection`: L1 coefficient magnitude at best CV alpha
  - `elasticnet_feature_selection`: L1+L2 (better than Lasso with correlated features, keeps groups)
  - `rfe_feature_selection`: Recursive Feature Elimination with RF (backward elimination, CV-optimal size)
  - `rf_feature_importance`: RF Gini impurity importance (summed across mean/std/slope)
  - `permutation_importance`: model-agnostic permutation importance on RF
  - `gradient_feature_selection`: LSTM input gradient magnitudes (3-seed, 50 epochs)
- Run all 6 methods, compare rankings, pick consensus top-K
- **`sffs_feature_selection`**: Sequential Forward Floating Selection (wrapper method).
  Directly optimizes mse on early-stop via RF. Greedy add-one + backtrack-remove.
  Searches over all 75 features. Slow (~1-2hr) but gives empirically optimal subset.
  Set `sffs_max_features` to cap subset size. Prints best subset and exits.
  Optional: set `sffs_start_features` to warm-start from a known good subset.

### NIG Hyperparameters (evidential mode, HIGH PRIORITY)
- λ₁ (evidence regularizer weight): {0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 1.0}
- These control the tradeoff between accuracy and uncertainty calibration
- Too high λ₁ → model becomes uncertain about everything
- Too low λ₁ → model is overconfident, uncertainty doesn't separate

### Difficulty Thresholds for 3-class conversion
- t₁ (easy/moderate boundary): try percentiles {10, 20, 25, 30, 33} of training miss_rate
- t₂ (moderate/hard boundary): try percentiles {67, 70, 75, 80, 85} of training miss_rate

### Architecture
- Hidden size: {32, 64, 128}
- Layers: {1, 2, 3}
- Dropout: {0.1, 0.2, 0.3, 0.4}

### Window & Horizon
- Start with window=30, horizon=30, sub_window=6 (n_steps=5)
- Also try horizon={10, 15, 20}, sub_window={3, 5, 10}
- train_stride=10, eval_stride=30

### Training
- LR: {1e-4, 5e-4, 1e-3, 2e-3}
- Weight decay: {0, 1e-5, 1e-4}
- Batch size: {32, 64, 128}
- Patience: {25, 40, 60}
- Gradient clipping: {0.5, 1.0, 2.0}

### Warmup Frames (low priority)
- warmup_frames: {0, 100, 200, 400} — test once, low priority

### Cross-camera generalization
- Feature set: original 65 vs all 75 (with new temporal/detector features)
- Cross-camera calibration: does fine-tuning on 30min of new camera help?
- Training already uses 4 cameras; try subsets (2 or 3) to measure impact

## Experiment Protocol

1. Read src/pod_train.py (or src/pod_train_evid.py) fully before starting
2. Run baseline first with default config. Record in the sweep's results TSV.
3. ONE change at a time
4. Run the experiment, parse the RESULT line
5. Log to the sweep's results TSV (results/pod_results_lstm.tsv or results/pod_results_evid.tsv)
6. Keep if: (a) mse_within improves, AND (b) mse_cross does not worsen by more than 10% relative to the best mse_cross seen so far, AND (c) cls_trans_within > 0.50. This cross-val guard prevents the search from drifting toward intersection memorization. For evidential mode additionally: unc_sep_within > 0.
7. Discard if: any keep condition is violated
8. Git commit each experiment
9. Crash → revert
10. NEVER STOP — continue until manually stopped

## Priority Order

### Sweep A (LSTM)
1. Establish baseline (LSTM, default config)
2. Sweep difficulty thresholds t₁, t₂
3. Architecture (hidden size, layers, dropout)
4. Training hyperparams (LR, batch size)
5. Feature subsets
6. Window/horizon/sub_window
7. Combine best settings

### Sweep B (Evidential)
1. Establish baseline (EvidentialLSTM, default config)
2. Sweep λ₁ (evidence regularizer): {0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 1.0}
3. Sweep difficulty thresholds t₁, t₂
4. Architecture (hidden size, layers, dropout)
5. Training hyperparams (LR, batch size)
6. Feature subsets
7. Window/horizon/sub_window
8. Combine best settings

### After Both Sweeps
- Run RF and LSTM+RF once at each sweep's optimal config for thesis comparison table

## CRASH SAFETY

Before each experiment:
1. git add -A && git commit -m "pre-exp NNN"
2. Edit pod_train.py
3. Run experiment
4. Append to the sweep's results TSV (pod_results_lstm.tsv or pod_results_evid.tsv)
5. git add -A && git commit -m "exp NNN: description — mse=X unc_sep=X cls_acc=X [keep/discard]"
6. If discard: git checkout HEAD~1 -- pod_train.py && git commit -m "revert exp NNN"

## Constraints

- Python 3 + PyTorch + scikit-learn + pandas + numpy + scipy
- Single GPU or CPU
- Do not modify CSV files
