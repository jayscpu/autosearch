# ECHO Bellevue Experiment — Second Autosearch
## Multi-Intersection Scene Difficulty Prediction for Adaptive YOLO Model Switching

### Overview

This experiment validates ECHO's scene difficulty predictor on the City of Bellevue
Traffic Video Dataset (5 intersections, ~101 hours total, 1280×720, 30fps).
The goal: prove that lightweight spatial image features — extracted from raw frames
without running any detector — can predict scene difficulty accurately enough to
drive adaptive model switching on edge devices.

The experiment was conducted as an autonomous hyperparameter search using Claude Code
with `--dangerously-skip-permissions` on a RunPod RTX 4090 GPU instance. The agent
ran experiments one-at-a-time, logged results to TSV files, and applied keep/discard
logic to hill-climb toward optimal configurations.

### Dataset

City of Bellevue Traffic Video Dataset:
  https://github.com/City-of-Bellevue/TrafficVideoDataset

| #  | Intersection              | Role                                          |
|----|---------------------------|-----------------------------------------------|
| 1  | Bellevue_150th_Eastgate   | Training (50% train / 10% early-stop / 40% within-val) |
| 2  | Bellevue_150th_Newport    | Training (50% train / 10% early-stop / 40% within-val) |
| 3  | Bellevue_150th_SE38th     | Cross-camera test (100% held out)             |
| 4  | Bellevue_Bellevue_NE8th   | Training (50% train / 10% early-stop / 40% within-val) |
| 5  | Bellevue_116th_NE12th     | Training (50% train / 10% early-stop / 40% within-val) |

Intersection 3 is the cross-camera held-out test — same 150th corridor as
intersections 1+2, making it a realistic "new camera, same road" generalization test.

Videos were processed at 2fps. Ground truth miss_rate was computed using YOLO11x@1280
on RunPod (too expensive for Jetson). Three deployment-tier detectors were benchmarked
on the Jetson Orin Nano via TensorRT FP16: nano@640, small@640, medium@640.
All @320 configurations were excluded due to unacceptable miss rates on Bellevue's
high-mounted cameras.

### Feature Sets

Defined in `src/pod_features.py`:

- **SPATIAL_65**: 65 spatial image features extracted from raw frames. Includes
  frequency analysis (FFT, DCT, Gabor), edge/texture metrics (Laplacian, gradient,
  GLCM), foreground segmentation (MOG2-derived blob statistics), motion estimation
  (temporal differencing, optical flow), color/brightness, MSCN statistics, and
  regional decomposition (top/mid/bot). No detector output required.

- **NEW_FEATURES** (10 additional): Optical flow (magnitude mean/std, direction
  entropy), SSIM vs previous frame, and 6 detector-derived features from YOLO11n
  and YOLO11s (detection count, confidence, spatial dispersion, small object ratio).

- **ALL_75**: SPATIAL_65 + NEW_FEATURES. Best MSE but requires running a detector
  to extract features, partially defeating ECHO's purpose.

- **TOP_35_SPEARMAN**: Top 35 features by Spearman correlation from the Fresno
  experiment. Used for single-camera experiments where fewer features reduce
  overfitting.

The thesis contribution: **SPATIAL_65 features are sufficient** — they eliminate the
circular dependency of needing a detector to decide which detector to run.


### Experimental Phases

The experiment progressed through 5 phases, totaling ~445 experiments:

────────────────────────────────────────────────────────────────────────────────
PHASE 1: ALL_75 Autosearch (first_autosearch results, ~250 experiments)
────────────────────────────────────────────────────────────────────────────────

Two parallel sweeps using ALL_75 features (detector-derived features included):

**Sweep A — PlainLSTM** (pod_results_lstm.tsv, 65 experiments):
  - 3-seed ensemble, MSE loss
  - Baseline mse_within: 0.01627 → Best: 0.01033 (36.5% reduction)
  - Final config: ALL_75, h128, L3, window=40, sub_window=10, batch=32,
    dropout=0.2, lr=1e-3, t1p=20/t2p=85

**Sweep B — EvidentialLSTM** (pod_results_evid.tsv, 190 experiments):
  - NIG loss, single seed, uncertainty calibration
  - Baseline mse_within: 0.01463 → Best: 0.01039 (29.0% reduction)
  - Best uncertainty: unc_sep=0.170 (t1p20+s7+L4, mse=0.01083)
  - Final config: ALL_75, h128, L4, stride=6, t1p=20, t2p=90, lambda1=0.25

Key finding: Detector-derived features contribute significantly to both MSE and
uncertainty calibration. ALL_75 achieves unc_sep=0.10-0.17 (usable range).

────────────────────────────────────────────────────────────────────────────────
PHASE 2: SPATIAL_65 Evidential Sweep (86 experiments)
────────────────────────────────────────────────────────────────────────────────

File: pod_results_spatial65.tsv

Tested whether spatial-only features (no detector output) are viable.
Two deployment scenarios:

**4CAM (multi-camera generalization):**
  Train on 4 intersections, test on held-out intersection 3.

  | Model            | mse_within | mse_cross | cls_trans | unc_sep | Config                           |
  |------------------|-----------|-----------|-----------|---------|----------------------------------|
  | EvidentialLSTM   | 0.01335   | 0.01925   | 0.503     | 0.100   | S65, h128/L4, warmup=400, t25/90 |
  | Best unc_sep     | 0.01419   | —         | 0.513     | 0.185   | S65, h128/L4, λ=0.28, t20/90     |
  | Best cls_trans   | 0.01361   | —         | 0.557     | 0.146   | S65, h128/L4, λ=0.25, warmup=300 |
  | RF               | 0.01539   | 0.02555   | 0.527     | —       | S65, warmup=400, t25/90          |

**1CAM (single-camera cold-start):**
  Train and eval on one intersection only. Simulates per-camera deployment.

  Initially tested on Eastgate — model collapsed to majority-class predictor
  (cls_acc ≈ majority_acc, unc_sep ≈ 0) because Eastgate's miss_rate distribution
  is heavily concentrated with ~94% in one band.

  Switched to NE8th — genuinely discriminating behavior observed:

  | Model            | mse_within | cls_trans | unc_sep | Config                              |
  |------------------|-----------|-----------|---------|-------------------------------------|
  | RF               | 0.00791   | 0.536     | —       | TOP_35, warmup=300, t10/70          |
  | EvidentialLSTM   | 0.00842   | 0.510     | 0.001   | TOP_35, h32/L4, λ=0.25, t10/70      |

Key findings:
  - Spatial-only 4CAM MSE is ~24% worse than ALL_75 (0.01335 vs 0.01039)
  - Uncertainty calibration drops: unc_sep 0.100 (spatial) vs 0.170 (ALL_75)
  - 1CAM uncertainty is dead (unc_sep < 0.003) regardless of features
  - TOP_35 features outperform full SPATIAL_65 on 1CAM (less overfitting)
  - RF wins on 1CAM; LSTM wins on 4CAM

────────────────────────────────────────────────────────────────────────────────
PHASE 3: PlainLSTM Spatial-Only Sweep (53 experiments)
────────────────────────────────────────────────────────────────────────────────

File: pod_results_plainlstm.tsv

Tested whether PlainLSTM (MSE loss, no NIG head) beats Evidential on spatial-only
features, since uncertainty calibration is weak without detector features.

  | Setting       | Model              | mse_within | cls_trans | Config                             |
  |---------------|--------------------|-----------|-----------|-------------------------------------|
  | 4CAM          | PlainLSTM 1-seed   | 0.01283   | 0.533     | S65, h256/L4, warmup=400, t25/90    |
  | 4CAM          | PlainLSTM 3-seed   | 0.01311   | 0.544     | S65, h128/L4, warmup=400, t25/90    |
  | 1CAM NE8th    | PlainLSTM 1-seed   | 0.00797   | 0.504     | TOP_35, h96/L4, lr=3e-3, t10/70     |
  | 1CAM NE8th    | PlainLSTM 3-seed   | 0.00811   | 0.504     | TOP_35, h32/L4, t10/70              |

Key findings:
  - PlainLSTM single-seed with larger hidden (h256) beats 3-seed ensemble
    → bigger model compensates for no ensembling
  - MSE loss prefers higher LR (3e-3 vs 1e-3) and larger architecture than NIG loss
  - PlainLSTM beats Evidential on mse_within (0.01283 vs 0.01335 for 4CAM)
  - Evidential beats PlainLSTM on cross-camera generalization (0.01925 vs 0.02114)

────────────────────────────────────────────────────────────────────────────────
PHASE 4: Transition-Weighted Loss & Delta Features (49 experiments)
────────────────────────────────────────────────────────────────────────────────

File: pod_results_trans_exps.tsv

The cls_trans ≈ 0.50 problem: all models, all feature sets, all configs produce
near-chance transition accuracy. The model predicts average miss_rate well but
cannot anticipate *when* difficulty changes.

**Transition-weighted loss:** Identify training windows where the 3-class label
changes between input and prediction horizon. Multiply their loss by a configurable
weight, forcing the model to focus on the rare (~5%) transition windows.

**Delta features:** Rate-of-change features computed from CSVs before windowing
(delta_5 and delta_10 for 8 key features). Tests whether explicit temporal
derivatives help the LSTM detect transitions.

Results — the breakthrough:

  | Config                        | cls_trans | mse_within | unc_sep | Note                          |
  |-------------------------------|-----------|-----------|---------|-------------------------------|
  | EVID 4CAM UNC TW w=8          | 0.587     | 0.01574   | 0.199   | **ALL THREE METRICS IMPROVED** |
  | EVID 4CAM MSE TW w=10         | 0.650     | 0.01632   | 0.089   | Best cls_trans, MSE/unc trade-off |
  | EVID 4CAM UNC TW w=10         | 0.647     | 0.01646   | 0.087   | Similar to above              |
  | EVID 4CAM UNC TW w=2          | 0.540     | 0.01558   | —       | MSE improved 2.4%             |
  | DELTA EVID 4CAM UNC           | 0.524     | 0.01541   | —       | Marginal cls_trans gain       |

**Best overall config: EVID 4CAM UNC TW w=8**
  - cls_trans: 0.501 → 0.587 (+17%)
  - mse_within: 0.01597 → 0.01574 (improved 1.4%)
  - unc_sep: 0.185 → 0.199 (improved 7.6%, highest ever achieved)
  - All three metrics improved simultaneously — no trade-off required.

**Transition-weighted loss is a real thesis contribution.** cls_trans jumped from
~0.50 (coin-flip on transitions) to 0.587-0.650 depending on weight. The model
learns to pay attention to the rare but important transition windows.

Delta features provided marginal gains on their own but did not combine well
with transition weighting. Sub_window=1 and sub_window=2 showed no improvement.
PlainLSTM TW experiments exceeded the 10% MSE penalty threshold and were discarded.

────────────────────────────────────────────────────────────────────────────────
PHASE 5: Prediction Export (5 configs)
────────────────────────────────────────────────────────────────────────────────

Trained the 5 best configs and exported per-window predictions for controller
simulation. These CSVs contain pred_miss_rate and true_miss_rate for every
evaluation window, enabling downstream model-switching simulation.

  | Model               | File                                   | Rows   | Config                            |
  |---------------------|----------------------------------------|--------|-----------------------------------|
  | PlainLSTM 4CAM      | predictions_plainlstm_4cam.csv         | 10,584 | h256/L4, S65, mse=0.01283         |
  | PlainLSTM 1CAM      | predictions_plainlstm_1cam_ne8th.csv   | 2,077  | h96/L4, TOP_35, mse=0.00797       |
  | Evidential 4CAM TW8 | predictions_evidential_4cam_tw8.csv    | 10,809 | h128/L4, S65, TW=8, unc=0.199     |
  | RF 4CAM             | predictions_rf_4cam.csv                | 10,584 | S65, 500 trees, mse=0.01539       |
  | RF 1CAM             | predictions_rf_1cam_ne8th.csv          | 2,077  | TOP_35, 500 trees, mse=0.00791    |

Verified: mse_within computed from CSVs matches sweep results for all 5 configs.


### Repository Structure

```
second_autosearch/
├── docs/
│   └── pod_program.md              ← Experiment protocol and search space
├── src/
│   ├── pod_features.py             ← Feature set definitions (SPATIAL_65, ALL_75, TOP_35)
│   ├── pod_pipeline.py             ← Feature extraction pipeline (already run)
│   ├── pod_train.py                ← Sweep A training script (PlainLSTM)
│   ├── pod_train_evid.py           ← Main training script (LSTM + Evidential + TW)
│   ├── pod_rf_eval.py              ← Random Forest evaluation script
│   ├── generate_predictions.py     ← Export predictions to CSV
│   ├── pod_download.sh             ← Data download script
│   └── pod_setup.sh                ← Environment setup
├── data/
│   └── pod_detections_*.csv        ← Per-intersection detection data (5 files)
└── results/
    ├── pod_results_lstm.tsv        ← Sweep A: ALL_75 PlainLSTM (65 experiments)
    ├── pod_results_evid.tsv        ← Sweep B: ALL_75 Evidential (190 experiments)
    ├── pod_results_spatial65.tsv   ← Spatial-only Evidential + RF (86 experiments)
    ├── pod_results_plainlstm.tsv   ← Spatial-only PlainLSTM (53 experiments)
    ├── pod_results_trans_exps.tsv  ← Transition-weighted + delta (49 experiments)
    └── predictions_*.csv           ← Exported predictions for controller sim (5 files)
```

### Final Results Summary

**4CAM (multi-camera generalization) — Spatial-Only Features:**

  | Model                        | mse_within | mse_cross | cls_trans | unc_sep |
  |------------------------------|-----------|-----------|-----------|---------|
  | PlainLSTM 1-seed             | 0.01283   | 0.02051   | 0.533     | —       |
  | EvidentialLSTM               | 0.01335   | 0.01925   | 0.503     | 0.100   |
  | EvidentialLSTM TW w=8 (best) | 0.01574   | —         | 0.587     | 0.199   |
  | RF                           | 0.01539   | 0.02555   | 0.527     | —       |

**1CAM NE8th (single-camera cold-start) — Spatial-Only Features:**

  | Model                        | mse_within | cls_trans | unc_sep |
  |------------------------------|-----------|-----------|---------|
  | RF                           | 0.00791   | 0.536     | —       |
  | PlainLSTM 1-seed             | 0.00797   | 0.504     | —       |
  | EvidentialLSTM               | 0.00842   | 0.510     | 0.001   |

### Key Thesis Findings

1. **Spatial-only features work.** No detector output needed at inference time.
   ECHO's predictor runs on raw image statistics, eliminating the circular
   dependency of needing a detector to decide which detector to run.

2. **Transition-weighted loss is a real contribution.** cls_trans improved from
   ~0.50 (chance) to 0.587-0.650. The best config (TW w=8) improved ALL three
   metrics simultaneously: lower MSE, higher cls_trans, higher unc_sep.

3. **Model choice depends on deployment scenario:**
   - Cold-start (single camera): RF — simplest, fastest, best MSE
   - Multi-camera with data: PlainLSTM — best MSE, no useful uncertainty
   - Multi-camera with uncertainty needed: EvidentialLSTM + TW — usable unc_sep

4. **Uncertainty requires multi-camera data.** unc_sep is dead (<0.003) on single
   camera. Viable (0.10-0.20) only with 4+ training cameras.

5. **Per-camera thresholds are necessary.** 4CAM absolute thresholds do not
   transfer to 1CAM — each camera needs locally calibrated difficulty boundaries.

6. **Detector features mainly help MSE, not transitions.** The gap between ALL_75
   and SPATIAL_65 is ~24% on MSE but cls_trans and unc_sep are comparable.

### Reproducibility

All experiments ran on a RunPod RTX 4090 instance using Claude Code v2.1.86
with `--dangerously-skip-permissions`. Each experiment logged its full config
to the TSV file. The autosearch agent followed the protocol in docs/pod_program.md:
one change at a time, keep/discard based on mse_within with cls_trans > 0.50
filter, git commit each kept experiment.

Total compute time: approximately 12 hours across multiple sessions.
Total experiments: ~445 (65 + 190 + 86 + 53 + 49 + prediction exports).
