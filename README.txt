
OVERVIEW: This repo contains two systematic hyperparameter searches for ECHO's scene difficulty predictor, using a 3-hour Fresno traffic camera recording (21,851 frames at 2fps, YOLO11n/m/x detections).
 
SEARCH 1: Binary Classification (110 experiments)
--------------------------------------------------
Goal: Predict whether YOLO11n@320 will fail (miss detections) in the next 15 seconds, using only spatial image features.
 
  Accuracy:       0.842
  Transition Acc: 0.641
  F1 Score:       0.898
  
Best config: LSTM+RF ensemble, top-35 Spearman features, miss_rate target, percentile-25 threshold, label smoothing=0.1
Note: Classification outputs a single binary prediction per 30-frame window. Not autoregressive — cannot be used with MPC.
 
 
SEARCH 2: Evidential Regression (98+ experiments)
--------------------------------------------------
Goal: Predict continuous miss_rate with calibrated uncertainty using Normal-Inverse-Gamma (NIG) evidential learning, outputting multi-step predictions for the MPC controller.
 
 2-Class Accuracy:       0.866 (peak) / 0.858 (stable)
  Uncertainty Separation: +0.006
  Transition Acc:         0.633
 
Best config: Evidential LSTM (NIG head), h=128, 2L, dropout=0.4, lambda1=0.3, lambda2=0.05, CosineAnnealingLR, multi-step (sub_window=6, 5 predictions), LayerNorm in head, top-35 Spearman features
Note: Outputs 5 sequential predictions with per-step uncertainty (autoregressive). Compatible with receding-horizon MPC.
 
 
FAIR COMPARISON (85 dedicated experiments)
------------------------------------------
 
All non-evidential models were tuned with equal effort (15-20 experiments each):
 
  Evidential NIG ........... acc=0.858   unc_sep=+0.006   (WINNER)
  XGBoost .................. acc=0.794   unc_sep= 0.000
  GradientBoosting ......... acc=0.794   unc_sep= 0.000
  GRU (tuned) .............. acc=0.794   unc_sep=-0.001
  RF (tuned) ............... acc=0.785   unc_sep=+0.001
  LSTM+RF Ensemble ......... acc=0.794   unc_sep=+0.001
  Plain LSTM (tuned) ....... acc=0.773   unc_sep=+0.001
 
Evidential NIG wins on both accuracy and uncertainty calibration.
 
 
KEY FINDINGS
------------
1. Percentile thresholding was the single biggest improvement (+0.12 accuracy)
2. Spearman feature selection (35 of 65 features) outperforms using all features
3. miss_rate (fn_nano/x_count) outperforms raw fn_nano as prediction target
4. Evidential NIG regression outperforms all classification and non-evidential approaches
5. Multi-step autoregressive prediction enables MPC-compatible output while improving uncertainty calibration
6. Evidential loss provides implicit regularization, yielding +8.5% accuracy over plain LSTM with identical architecture (????)
7. Uncertainty calibration is small but uniquely positive — NIG is the only approach achieving unc_sep > 0
 
 
REPO STRUCTURE
--------------
  autosearch/
  |-- README.md
  |-- program.md                    Classification search instructions
  |-- train.py                      Classification training script (final state)
  |-- results.tsv                   110 classification experiments
  |-- yolox_pipeline.py             Generates the CSV data from raw video
  |-- yolox_features.csv            21,851 frames x 65 spatial features
  |-- yolox_detections.csv          Frame-level YOLO detection results
  |
  +-- evidential/                   Evidential regression search
      |-- program_reg.md            Regression search instructions
      |-- train_reg.py              Evidential LSTM (final state)
      |-- evidential_results.tsv    98+ evidential experiments
      |-- comparisons/              Fair comparison scripts (Extra runs for non evidential)
      |   |-- evidential_compare.py
      |   +-- evidential_fair_sweep.py
      +-- logs/                     Training logs
          +-- stderr_*.log
 
 
DATA PIPELINE
-------------
yolox_pipeline.py generates all data from a 3-hour Fresno traffic camera recording:
  1. Extract frames at 2fps (960x540) via ffmpeg
  2. Run YOLO11n@320, YOLO11m@640, YOLO11x@640 (pseudo ground truth)
  3. Greedy IoU matching -> fn_nano, fn_medium per frame
  4. Extract 65 spatial features per frame (edges, texture, frequency, density, BRISQUE)
  5. Output: yolox_features.csv + yolox_detections.csv
 
 
METHODOLOGY
-----------
Both searches follow the autoresearch pattern:
  - AI agent (Claude Code) edits a single training script
  - Each experiment changes ONE hyperparameter
  - Results are logged, kept or discarded based on metrics
  - Git commit after every experiment for crash recovery
  - Agent runs autonomously until stopped
 
Classification search: ~15 hours, 110 experiments
Evidential search: ~12 hours, 98+ experiments + 85 fair comparison experiments
 
 
