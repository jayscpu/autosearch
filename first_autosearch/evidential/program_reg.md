# ECHO Evidential Regression — Autonomous Search

## Context

You are optimizing an Evidential LSTM that predicts continuous miss_rate (fn_nano / x_count) using Normal-Inverse-Gamma (NIG) uncertainty quantification. This is for ECHO's full pipeline: the NIG output feeds into an MPC controller via Student-t CDF conversion to class probabilities P(easy), P(moderate), P(hard).

The previous autosearch (classification) achieved 84.2% binary accuracy. Now we need the evidential regression to produce:
1. Accurate continuous miss_rate predictions
2. Calibrated aleatoric/epistemic uncertainty (uncertainty should be HIGHER for incorrect predictions)
3. Class probabilities via Student-t CDF that the MPC can consume

The data is pre-extracted: ~21,851 frames from a 3-hour Fresno traffic camera at 2fps.

## Repo Structure

```
autosearch_regression/
├── program.md              ← YOU ARE HERE (human edits this)
├── train_reg.py            ← THE FILE YOU EDIT (agent edits this)
├── results.tsv             ← experiment log (agent appends here)
├── yolox_features.csv      ← pre-extracted features (DO NOT MODIFY)
├── yolox_detections.csv    ← detection results (DO NOT MODIFY)
```

## Fixed Files (DO NOT MODIFY)

- `yolox_features.csv` — 21,851 rows with 65 spatial features
- `yolox_detections.csv` — frame-level detection counts

## The File You Edit: train_reg.py

`train_reg.py` is a self-contained script that:
1. Loads pre-computed features
2. Computes continuous miss_rate target (fn_nano / x_count)
3. Builds shifted-label windows (features from current window, target = mean miss_rate over horizon)
4. Trains an Evidential LSTM with NIG output (γ, ν, α, β)
5. Evaluates prediction quality AND uncertainty calibration
6. Converts predictions to 3-class probabilities via Student-t CDF
7. Prints results

The script must print exactly one results line to stdout:
```
RESULT\tmse=X.XXX\tmae=X.XXX\tunc_separation=X.XXX\tcls_acc=X.XXX\tcls_trans_acc=X.XXX\tconfig=<json>
```

## Primary Metrics

**Three goals, all must be good:**

1. **Prediction accuracy:** Minimize MSE and MAE on continuous miss_rate prediction
2. **Uncertainty calibration:** Maximize `unc_separation` — the difference in mean epistemic uncertainty between incorrect and correct 3-class predictions. MUST be > 0 (uncertain predictions should have higher epistemic uncertainty). This is what failed with binary Dirichlet.
3. **Classification quality:** `cls_acc` > 0.70 and `cls_trans_acc` > 0.50 on 3-class predictions derived from Student-t CDF

The ranking: first filter by unc_separation > 0, then maximize cls_acc, then minimize MSE.

## NIG Architecture

The LSTM outputs 4 parameters per prediction:
- γ (gamma): predicted miss_rate mean
- ν (nu): precision of mean estimate (confidence), must be > 0
- α (alpha): shape parameter, must be > 1
- β (beta): scale parameter, must be > 0

Uncertainty decomposition:
- Aleatoric: β / (α - 1)
- Epistemic: β / (ν × (α - 1))

Training loss (from Amini et al. 2020):
- L_NLL: negative log-likelihood of Student-t marginal
- L_R: evidence regularizer |y - γ| × (2ν + α) — penalizes confident wrong predictions
- L_U: uncertainty regularizer for stable gradients when α → 1
- L_total = L_NLL + λ₁ × L_R + λ₂ × L_U

Student-t CDF conversion to 3 classes:
- Degrees of freedom: 2α
- Scale: sqrt(β(1+ν) / (ν × α))
- P(easy) = F(t₁), P(moderate) = F(t₂) - F(t₁), P(hard) = 1 - F(t₂)
- t₁, t₂ are difficulty thresholds (searchable)

## Search Space

### Features (use classification results as starting point)
- Start with top-35 Spearman features from classification search (proven best)
- Also try all 65, top-20, top-25, top-30, top-40
- The optimal subset for regression may differ from classification

### NIG Hyperparameters (HIGH PRIORITY — this is new)
- λ₁ (evidence regularizer weight): {0.01, 0.05, 0.1, 0.25, 0.5, 1.0}
- λ₂ (uncertainty regularizer weight): {0.001, 0.01, 0.05, 0.1}
- These control the tradeoff between accuracy and uncertainty calibration
- Too high λ₁ → model becomes uncertain about everything
- Too low λ₁ → model is overconfident, uncertainty doesn't separate

### Difficulty Thresholds for 3-class conversion
- t₁ (easy/moderate boundary): try percentiles {20, 25, 30, 33, 40} of training miss_rate
- t₂ (moderate/hard boundary): try percentiles {60, 67, 70, 75, 80} of training miss_rate
- These determine the 3-class labels for cls_acc evaluation

### Architecture
- Hidden size: {32, 64, 128}
- Layers: {1, 2, 3}
- Dropout: {0.1, 0.2, 0.3, 0.4}
- The output layer must enforce: ν > 0, α > 1, β > 0
  - Use softplus for all, add 1 to α output

### Window & Horizon
- Start with window=30, horizon=30 (proven best in classification)
- Also try horizon={10, 15, 20} — shorter horizons may be easier for regression
- train_stride=10, eval_stride=30

### Training
- LR: {1e-4, 5e-4, 1e-3, 2e-3}
- Weight decay: {0, 1e-5, 1e-4}
- Batch size: {32, 64, 128}
- Patience: {25, 40, 60} — regression may need longer training
- Gradient clipping: {0.5, 1.0, 2.0}

## Experiment Protocol

1. Read train_reg.py fully before starting.
2. Run baseline first. Record in results.tsv.
3. Change ONE thing at a time.
4. Run the experiment (should take 1-5 minutes).
5. Parse the RESULT line.
6. Log to results.tsv:

```
commit	mse	mae	unc_sep	cls_acc	cls_trans	status	description
a1b2c3d	0.045	0.15	0.020	0.680	0.520	baseline	35 feat, h=64, 2L
b2c3d4e	0.042	0.14	0.035	0.700	0.540	keep	λ₁=0.1
```

7. Keep if: unc_separation > 0 AND cls_acc improves (or MSE improves with unc_sep maintained)
8. Discard if: unc_separation ≤ 0 or cls_acc drops significantly
9. Crash → revert

## Priority Order

1. Establish baseline with top-35 features, default NIG params
2. Sweep λ₁ (evidence regularizer) — this is the MOST important for uncertainty calibration
3. Sweep λ₂ (uncertainty regularizer)
4. Sweep difficulty thresholds t₁, t₂
5. Architecture (hidden size, layers, dropout)
6. Training hyperparams (LR, batch size)
7. Feature subsets
8. Window/horizon
9. Combine best settings

## NEVER STOP

Once the experiment loop begins, do NOT pause. Continue indefinitely until manually stopped.

## CRASH SAFETY

Before each experiment:
1. git add -A && git commit -m "pre-exp NNN"
2. Edit train_reg.py
3. Run experiment
4. Append to results.tsv
5. git add -A && git commit -m "exp NNN: description — mse=X unc_sep=X cls_acc=X [keep/discard]"
6. If discard: git checkout HEAD~1 -- train_reg.py && git commit -m "revert exp NNN"

## Constraints

- Python 3 + PyTorch + scikit-learn + pandas + numpy + scipy
- Single GPU or CPU
- Each experiment under 5 minutes
- Do not modify CSV files
- Temporal split at frame_id = 14400
- Val windows non-overlapping (eval_stride = window)
