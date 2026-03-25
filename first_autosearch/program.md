# ECHO Fresno Experiment — Autonomous Search

## Context

You are optimizing a neural network pipeline that predicts when a lightweight object detector (YOLO11n@320) will fail on a traffic camera stream, using only image-derived spatial features. This is for a thesis project called ECHO (Energy-Constrained Hierarchical Orchestrator).

The data is pre-extracted: ~21,851 frames from a 3-hour Fresno traffic camera recording at 2fps. For each frame, 65 spatial features and detection results (fn_nano, fn_medium, x_count, nano_tp, nano_fp, etc.) are already computed. You do NOT need to re-extract features — work only with the existing CSVs.

## Repo Structure

```
fresno_experiment/yolox/
├── program.md              ← YOU ARE HERE (human edits this)
├── train.py                ← THE FILE YOU EDIT (agent edits this)
├── results.tsv             ← experiment log (agent appends here)
├── yolox_features.csv      ← pre-extracted features (DO NOT MODIFY)
├── yolox_detections.csv    ← detection results (DO NOT MODIFY)
```

## Fixed Files (DO NOT MODIFY)

- `yolox_features.csv` — 21,851 rows. Columns: sequence, frame_id, fn_nano, fn_medium, x_count, nano_tp, nano_fp, medium_tp, medium_fp, + 65 spatial features
- `yolox_detections.csv` — frame-level detection counts and IoU results

## The File You Edit: train.py

`train.py` is a self-contained script that:
1. Loads `yolox_features.csv`
2. Defines configuration (features, window, horizon, model, hyperparams, target)
3. Builds windows with temporal split (train: frame_id ≤ 14400, val: frame_id > 14400)
4. Trains a model
5. Evaluates and prints a single result line

The script must print exactly one results line to stdout in this format:
```
RESULT\tacc=0.XXX\ttrans_acc=0.XXX\tf1=0.XXX\tprec=0.XXX\trec=0.XXX\tn_val=XXX\tconfig=<json>
```

## Primary Metric

**Maximize `acc` (validation accuracy) on fn_nano prediction**, subject to:
- `trans_acc` must be > 0.50 (must beat random chance on transitions)
- Must beat persistence baseline (acc ≈ 0.628)

Secondary objectives (break ties): higher `trans_acc`, then higher `f1`.

## Current Best Baseline

```
acc=0.632  trans_acc=0.609  f1=0.676  (Dirichlet LSTM, 20 features, window=30, horizon=30)
acc=0.603  trans_acc=0.609  f1=0.660  (RF Classifier, same config)
```

Beat these.

## Search Space

You may modify ANY of the following in train.py. Change ONE thing at a time to isolate effects.

### Features (high priority)
The current pipeline uses only 20 of the 65 available features. All 65 are in the CSV:

**Currently selected (20):**
fast_keypoints_half, edge_density_coarse, gradient_magnitude_mean, gabor_nyquist_energy, fast_keypoints_full, glcm_homogeneity, overall_edge_density, top_edge_density, top_high_freq_energy, top_gradient_std, fft_critical_band_ratio, total_edge_components, top_laplacian_var, glcm_entropy, mid_edge_density, dark_channel_mean, mid_gradient_std, foreground_blob_count, mid_high_freq_energy, rms_contrast

**Available but unused (45):**
fft_high_freq_energy_ratio, laplacian_variance, laplacian_mean_abs, edge_density_fine, edge_fine_coarse_ratio, keypoint_loss_ratio, spatial_frequency, gradient_magnitude_std, dct_high_freq_ratio, downsample_ssim, downsample_info_loss, foreground_pixel_ratio, shadow_pixel_ratio, foreground_blob_area_mean, foreground_blob_area_std, foreground_blob_pa_ratio, foreground_edge_density, temporal_diff_mean, temporal_diff_std, motion_pixel_ratio, image_entropy, glcm_contrast, glcm_energy, colorfulness, mean_brightness, brightness_std, small_edge_components, saturation_mean, saturation_std, mscn_mean, mscn_std, mscn_kurtosis, mscn_skewness, mscn_h_pair_mean, mscn_h_pair_std, mscn_v_pair_mean, mscn_v_pair_std, mid_laplacian_var, bot_laplacian_var, bot_gradient_std, bot_edge_density, bot_high_freq_energy, ratio_top_bot_edge_density, ratio_top_bot_gradient_std, ratio_top_bot_laplacian_var

Feature search directions:
- Try all 65 features
- Try subsets based on correlation with target
- Try removing features that add noise
- Note: `foreground_blob_area_mean`, `temporal_diff_mean`, `motion_pixel_ratio` were filtered by DETRAC-based selection but may help on Fresno

### Target Definition (high priority)
Current target: binary label based on raw `fn_nano >= threshold` in the horizon window.

Try instead:
- **Miss rate**: `fn_nano / x_count` — normalizes by scene density
- **Per-frame F1**: `2*nano_tp / (2*nano_tp + nano_fp + fn_nano)` — accounts for both false positives and false negatives
- **Different thresholding**: percentile-based, fixed values, balanced split
- The detections CSV has all columns needed: fn_nano, nano_tp, nano_fp, x_count

### Window & Horizon (high priority)
Current: WINDOW=30, HORIZON=30 (15 seconds each at 2fps)

Try:
- WINDOW: {10, 15, 20, 25, 30, 40, 50}
- HORIZON: {5, 10, 15, 20, 30, 45, 60}
- Shorter horizons may be easier to predict
- Longer windows may give more context
- TRAIN_STRIDE: {10, 15, 30} (affects training set size/correlation)
- EVAL_STRIDE should equal WINDOW (non-overlapping val)

### Model Architecture (medium priority)
Current: 2-layer Dirichlet LSTM with evidential loss.

Try:
- **Plain LSTM + softmax + cross-entropy** (drop Dirichlet head entirely — priority change)
- GRU instead of LSTM
- 1 layer vs 2 layers vs 3 layers
- Hidden size: {32, 64, 128, 256}
- Dropout: {0.1, 0.2, 0.3, 0.4}
- RF alone (sometimes RF beats LSTM on small data)
- RF hyperparams: n_estimators, max_depth, min_samples_leaf

### Training Hyperparameters (lower priority)
- Learning rate: {5e-4, 1e-3, 2e-3, 5e-3}
- Weight decay: {0, 1e-5, 1e-4, 1e-3}
- Batch size: {32, 64, 128}
- Early stopping patience: {15, 25, 40}
- Gradient clipping: {0.5, 1.0, 2.0}

### RF Feature Engineering (medium priority)
Current: mean, std, slope per feature across window → 60 RF features.

Try:
- Add min/max per feature
- Add last-value (most recent frame in window)
- Add first-to-last difference
- Percentile features (25th, 75th)

## Experiment Protocol

1. Read train.py fully before starting.
2. Run the current baseline first. Record it in results.tsv.
3. Change ONE thing at a time.
4. Run the experiment (should take 1-3 minutes).
5. Parse the RESULT line from stdout.
6. Log to results.tsv:

```
commit	acc	trans_acc	f1	status	description
a1b2c3d	0.632	0.609	0.676	baseline	original 20 features, LSTM, w=30 h=30
b2c3d4e	0.641	0.620	0.685	keep	all 65 features
c3d4e5f	0.625	0.580	0.670	discard	GRU instead of LSTM
```

7. If `acc` improves AND `trans_acc` > 0.50 → keep the change (status=keep)
8. If `acc` drops or `trans_acc` ≤ 0.50 → revert (status=discard)
9. If the run crashes → revert (status=crash), note the error
10. Commit each experiment to git: `git add -A && git commit -m "<description>"`
11. Continue to next experiment.

## Priority Order

Try changes roughly in this order (highest impact first):
1. Drop Dirichlet → plain softmax cross-entropy (keep everything else the same)
2. Switch target: fn_nano → miss rate (fn_nano / x_count)
3. Use all 65 features instead of 20
4. Sweep WINDOW: {15, 20, 30, 40}
5. Sweep HORIZON: {10, 15, 30}
6. Feature selection: run Spearman correlation on Fresno data, pick top-k
7. Model architecture: GRU, 1-layer, hidden size
8. RF hyperparameter tuning
9. Training hyperparameters (LR, batch size)
10. RF feature engineering (add min/max/percentiles)

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working indefinitely until manually stopped.

If you run out of ideas:
- Re-read the search space above for unexplored directions
- Try combining two individually successful changes
- Try more radical changes (e.g., window=10, completely different feature set)
- Try the opposite of what's currently working (if big windows help, try even bigger)
- Check if individual promising features can be combined with the best model

## Constraints

- Python 3 + PyTorch + scikit-learn + pandas + numpy (all already installed)
- Single GPU (or CPU if GPU unavailable)
- Each experiment should complete in under 5 minutes
- Do not modify the CSV data files
- Always use temporal split at frame_id = 14400 (first 2h train, last 1h val)
- Val windows must be non-overlapping (EVAL_STRIDE = WINDOW)
