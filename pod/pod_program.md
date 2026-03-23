# ECHO Bellevue Experiment — Autonomous Search (Regression)

## Context

Multi-intersection generalization experiment for ECHO. Uses 5 Bellevue traffic
camera intersections (~101 hours total). Tests whether spatial features trained
on 4 intersections generalize to unseen intersection 3.

Every run trains 4 models and picks the best by `mse_within`:
1. **LSTM** — MSE loss, multi-seed ensemble (3 seeds)
2. **RF** — RandomForestRegressor on summary features (mean/std/slope)
3. **LSTM+RF** — averaged predictions from 1 and 2
4. **EvidentialLSTM** — NIG loss with uncertainty quantification, Student-t CDF
   conversion to 3-class probabilities for the MPC controller

Data is pre-extracted by `pod_pipeline.py`. Work only with the CSVs.

## Repo Structure

```
autosearch/pod/
├── pod_program.md          ← YOU ARE HERE
├── pod_train.py            ← THE FILE YOU EDIT
├── pod_results.tsv         ← experiment log
├── pod_features.py         ← shared feature list
├── pod_pipeline.py         ← extraction pipeline (already run)
├── pod_features_all.csv    ← combined features (DO NOT MODIFY)
├── pod_features_{intersection}.csv  ← per-intersection features
└── pod_detections_{intersection}.csv
```

## Experimental Design

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
- All 4 models run every experiment — no mode switching
- The best model is selected automatically by mse_within
- Hyperparameter changes affect all models equally (architecture, features, etc.)
- λ₁ only affects EvidentialLSTM; RF params only affect RF

### Features
- Start with top-35 Spearman features from classification search (proven best)
- Also try all 65, all 75, top-20, top-25, top-30, top-40
- The optimal subset for regression may differ from classification

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

### Cross-camera generalization
- Feature set: original 65 vs all 75 (with new temporal/detector features)
- Cross-camera calibration: does fine-tuning on 30min of new camera help?
- Training already uses 4 cameras; try subsets (2 or 3) to measure impact

## Experiment Protocol

1. Read pod_train.py fully before starting
2. Run baseline first (all 4 models train every run). Record in pod_results.tsv.
3. ONE change at a time
4. Run the experiment, parse the RESULT line
5. Log to pod_results.tsv
6. Keep if: mse_within improves AND cls_trans > 0.50
7. Discard if: mse_within worsens or cls_trans drops below 0.50
8. Git commit each experiment
9. Crash → revert
10. NEVER STOP — continue until manually stopped

## Priority Order

1. Establish baseline (all 4 models run, default config)
2. Sweep λ₁ (evidence regularizer): {0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 1.0}
3. Sweep difficulty thresholds t₁, t₂
4. Architecture (hidden size, layers, dropout)
5. Training hyperparams (LR, batch size)
6. Feature subsets
7. Window/horizon/sub_window
8. RF hyperparams (n_estimators, max_depth, min_samples_leaf)
9. Combine best settings

## CRASH SAFETY

Before each experiment:
1. git add -A && git commit -m "pre-exp NNN"
2. Edit pod_train.py
3. Run experiment
4. Append to pod_results.tsv
5. git add -A && git commit -m "exp NNN: description — mse=X unc_sep=X cls_acc=X [keep/discard]"
6. If discard: git checkout HEAD~1 -- pod_train.py && git commit -m "revert exp NNN"

## Constraints

- Python 3 + PyTorch + scikit-learn + pandas + numpy + scipy
- Single GPU or CPU
- Each experiment under 5 minutes
- Do not modify CSV files
