#!/usr/bin/env python3
"""ECHO Bellevue — Multi-intersection regression + evidential regression.

Loads pre-extracted features, trains on intersections 1, 2, 4, 5, evaluates on:
  1. Within-camera val (temporal split of train intersections)
  2. Cross-camera val (unseen intersection 3)
  3. Combined val

Two-sweep design:
  --mode lstm:       trains PlainLSTM only (3-seed ensemble, MSE loss)
  --mode evidential: trains EvidentialLSTM only (NIG loss, single seed)

Prints RESULT line parsed by the agent.
"""

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pod_features import ALL_FEATURES, TOP_35_SPEARMAN

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Mode ── ("lstm" or "evidential")
    "mode": "evidential",

    # ── Features ──
    "features": TOP_35_SPEARMAN,

    # ── Target ──
    "target": "miss_rate",

    # ── Windowing ──
    "window": 30,
    "horizon": 30,
    "sub_window": 6,          # multi-step: each step predicts mean over sub_window frames
    "train_stride": 10,
    "eval_stride": 30,
    "warmup_frames": 200,      # skip first N frames per video (MOG2 bg model warm-up)

    # ── Intersections ──
    "train_intersections": [
        "Bellevue_150th_Eastgate",
        "Bellevue_150th_Newport",
        "Bellevue_Bellevue_NE8th",
        "Bellevue_116th_NE12th",
    ],
    "test_intersection": "Bellevue_150th_SE38th",
    "train_fraction": 0.50,
    "earlystop_fraction": 0.60,   # cumulative: 50-60% = early-stop, 60-100% = within-val

    # ── NIG Loss Hyperparameters (EvidentialLSTM) ──
    "lambda1": 0.3,           # evidence regularizer weight

    # ── Difficulty Thresholds (percentiles of training miss_rate) ──
    "t1_percentile": 10,      # easy/moderate boundary
    "t2_percentile": 85,      # moderate/hard boundary

    # ── Architecture (shared by LSTM and EvidentialLSTM) ──
    "hidden_size": 128,
    "n_layers": 2,
    "dropout": 0.4,

    # ── Training ──
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 300,
    "patience": 40,
    "grad_clip": 1.0,
    "seeds": [42, 123, 456],  # LSTM ensemble seeds

    # ── RF ──
    "rf_n_estimators": 500,
    "rf_max_depth": 20,
    "rf_min_samples_leaf": 10,

    # ── Feature selection (set True to print rankings, then continue training) ──
    "spearman_feature_selection": False,
    "mi_feature_selection": False,
    "lasso_feature_selection": False,
    "elasticnet_feature_selection": False,
    "rfe_feature_selection": False,
    "rf_feature_importance": False,
    "permutation_importance": False,
    "gradient_feature_selection": False,
    "sffs_feature_selection": False,  # Sequential Forward Floating Selection (slow, ~1-2hr)
    "sffs_max_features": 50,          # stop SFFS after this many features
    "sffs_start_features": None,      # list of features to start from (None = empty)
}

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FEATURES_CSV = DATA_DIR / "pod_features_all.csv"


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(FEATURES_CSV)
    if CONFIG["target"] == "miss_rate":
        # x_count=0 → miss_rate=0 (no vehicles to miss); clip prevents div-by-zero
        df["miss_rate"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    return df


def get_target_column():
    t = CONFIG["target"]
    if t == "miss_rate":
        return "miss_rate"
    return t


# ═══════════════════════════════════════════════════════════════════
# SPLITTING
# ═══════════════════════════════════════════════════════════════════

def split_data(df):
    """Split into train, early-stop, within-camera val, and cross-camera val.

    Per training intersection (temporally ordered):
      [0 .. 50%)   → train
      [50% .. 60%)  → early-stop (used only for checkpoint selection)
      [60% .. 100%] → within-camera val (never seen during training)
    Held-out intersection → cross-camera val.
    """
    train_ints = CONFIG["train_intersections"]
    test_int = CONFIG["test_intersection"]

    train_dfs, earlystop_dfs, within_val_dfs = [], [], []
    for int_name in train_ints:
        sub = df[df["sequence"] == int_name].sort_values("frame_id")
        n = len(sub)
        train_end = int(n * CONFIG["train_fraction"])
        earlystop_end = int(n * CONFIG["earlystop_fraction"])

        train_dfs.append(sub.iloc[:train_end])
        earlystop_dfs.append(sub.iloc[train_end:earlystop_end])
        within_val_dfs.append(sub.iloc[earlystop_end:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    earlystop_df = pd.concat(earlystop_dfs, ignore_index=True)
    within_val_df = pd.concat(within_val_dfs, ignore_index=True)
    cross_val_df = df[df["sequence"] == test_int].copy()

    return train_df, earlystop_df, within_val_df, cross_val_df


# ═══════════════════════════════════════════════════════════════════
# WINDOWING (multi-step: N_steps targets per sample)
# ═══════════════════════════════════════════════════════════════════

def build_windows(df, feature_cols, stride, scaler):
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S
    total_need = W + H
    target_col = get_target_column()

    X_list, y_list = [], []
    seq_labels = []
    # Track the first index of each video group so transition_accuracy can
    # skip comparisons at video boundaries (those pairs aren't temporal neighbours)
    boundary_indices = set()

    # Group by sequence AND video to avoid windows spanning video boundaries
    # (temporal features like optical flow reset to 0 at each video start)
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = CONFIG.get("warmup_frames", 0)
    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        # Skip MOG2 warm-up frames: foreground_pixel_ratio and foreground_blob_count
        # are systematically inflated before the background model converges (~200 frames)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        # Mark the start of this group as a boundary (except the very first group)
        if len(X_list) > 0:
            boundary_indices.add(len(X_list))

        seq_name = sdf["sequence"].iloc[0]
        feat_vals = scaler.transform(sdf[feature_cols].values)
        target_vals = sdf[target_col].values

        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t:t + W])

            # Multi-step targets: mean miss_rate for each sub-window
            targets = np.zeros(n_steps, dtype=np.float32)
            for step in range(n_steps):
                start = t + W + step * S
                end = start + S
                targets[step] = target_vals[start:end].mean()

            y_list.append(targets)
            seq_labels.append(seq_name)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)  # shape: (N, n_steps)
    return X, y, boundary_indices, seq_labels


# ═══════════════════════════════════════════════════════════════════
# FEATURE SELECTION (print rankings, continue with normal training)
# ═══════════════════════════════════════════════════════════════════

def _print_ranking(title, ranked):
    """Print a feature ranking to stderr."""
    print(f"\n=== {title} ===", file=sys.stderr)
    for rank, (fname, score) in enumerate(ranked, 1):
        print(f"  {rank:3d}. {fname:40s}  {score:.6f}", file=sys.stderr)
    print(f"=== END {title} ===\n", file=sys.stderr)


def run_feature_selection(train_df, feature_cols, X_train, y_train, n_feat, device):
    """Run any enabled feature selection methods. Prints rankings to stderr."""
    warmup = CONFIG.get("warmup_frames", 0)
    target_col = get_target_column()

    # Frame-level training data (warmup-skipped) for correlation-based methods
    def _get_frame_data():
        group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
        kept = []
        for _, sdf in train_df.groupby(group_cols):
            sdf = sdf.sort_values("frame_id")
            if warmup > 0:
                sdf = sdf.iloc[warmup:]
            kept.append(sdf)
        frame_df = pd.concat(kept, ignore_index=True)
        return frame_df

    # ── Spearman correlation ──
    if CONFIG.get("spearman_feature_selection"):
        frame_df = _get_frame_data()
        y_frames = frame_df[target_col].values
        scores = []
        for f in feature_cols:
            rho, _ = spearmanr(frame_df[f].values, y_frames)
            scores.append((f, abs(rho) if not np.isnan(rho) else 0.0))
        scores.sort(key=lambda x: -x[1])
        _print_ranking("SPEARMAN FEATURE RANKING", scores)

    # ── Mutual Information ──
    if CONFIG.get("mi_feature_selection"):
        X_train_rf = build_rf_features(X_train)
        n_raw = n_feat
        mi_scores = np.zeros(n_raw)
        for i in range(n_raw):
            feat_block = X_train_rf[:, [i, i + n_raw, i + 2 * n_raw]]
            mi = mutual_info_regression(feat_block, y_train.mean(axis=1),
                                        n_neighbors=5, random_state=42)
            mi_scores[i] = mi.sum()
        ranked = sorted(zip(feature_cols, mi_scores), key=lambda x: -x[1])
        _print_ranking("MUTUAL INFORMATION FEATURE RANKING", ranked)

    # ── Lasso (L1 regularization) ──
    # NOTE: Lasso and ElasticNet fit their own scaler on warmup-skipped
    # frame-level data. The main pipeline fits a different scaler on the full
    # train_df (no frame-level warmup skip). Coefficient magnitudes reflect a
    # slightly different standardization than what the LSTM sees, so these
    # rankings are not directly comparable to the windowed methods (MI, RF
    # importance, gradient). Use as a cross-check, not as ground truth.
    if CONFIG.get("lasso_feature_selection"):
        frame_df = _get_frame_data()
        y_frames = frame_df[target_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(frame_df[feature_cols].values)
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y_frames)
        coefs = np.abs(lasso.coef_)
        print(f"  Lasso best alpha={lasso.alpha_:.6f}, "
              f"non-zero={np.sum(coefs > 0)}/{n_feat}", file=sys.stderr)
        ranked = sorted(zip(feature_cols, coefs), key=lambda x: -x[1])
        _print_ranking("LASSO FEATURE RANKING", ranked)

    # ── Elastic Net (L1 + L2, better with correlated features) ──
    if CONFIG.get("elasticnet_feature_selection"):
        frame_df = _get_frame_data()
        y_frames = frame_df[target_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(frame_df[feature_cols].values)
        enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                            cv=5, random_state=42, max_iter=10000)
        enet.fit(X_scaled, y_frames)
        coefs = np.abs(enet.coef_)
        print(f"  ElasticNet best alpha={enet.alpha_:.6f}, "
              f"l1_ratio={enet.l1_ratio_:.2f}, "
              f"non-zero={np.sum(coefs > 0)}/{n_feat}", file=sys.stderr)
        ranked = sorted(zip(feature_cols, coefs), key=lambda x: -x[1])
        _print_ranking("ELASTIC NET FEATURE RANKING", ranked)

    # ── RFE (Recursive Feature Elimination with RF, backward elimination) ──
    if CONFIG.get("rfe_feature_selection"):
        X_train_rf = build_rf_features(X_train)
        y_train_mean = y_train.mean(axis=1)
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=42)
        # RFECV with 3-fold CV to find optimal number of features
        # step=3 removes 3 RF-summary features per iteration (= 1 raw feature,
        # since each raw feature maps to 3 summary stats: mean/std/slope)
        rfe = RFECV(rf, step=3, cv=3, scoring="neg_mean_squared_error",
                    min_features_to_select=10, n_jobs=-1)
        rfe.fit(X_train_rf, y_train_mean)
        # Map RFE ranking back to raw features: min rank across mean/std/slope
        n_raw = n_feat
        raw_ranks = np.zeros(n_raw)
        for i in range(n_raw):
            raw_ranks[i] = min(rfe.ranking_[i],
                               rfe.ranking_[i + n_raw],
                               rfe.ranking_[i + 2 * n_raw])
        n_selected = np.sum(raw_ranks == 1)
        print(f"  RFE optimal: {n_selected} raw features "
              f"({rfe.n_features_} RF-summary features)", file=sys.stderr)
        # Score = negative rank (rank 1 = best → highest score)
        ranked = sorted(zip(feature_cols, -raw_ranks), key=lambda x: -x[1])
        # Print with actual rank values
        ranked_for_print = [(f, -s) for f, s in ranked]
        _print_ranking("RFE FEATURE RANKING (lower rank = better)", ranked_for_print)
        # Also print the selected subset
        selected = [f for f, r in zip(feature_cols, raw_ranks) if r == 1]
        print(f"\n  RFE selected features ({len(selected)}):", file=sys.stderr)
        for f in selected:
            print(f"    {f}", file=sys.stderr)

    # ── RF impurity importance ──
    if CONFIG.get("rf_feature_importance"):
        X_train_rf = build_rf_features(X_train)
        y_train_mean = y_train.mean(axis=1)
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=42)
        rf.fit(X_train_rf, y_train_mean)
        # Map back to raw features: sum importance of mean/std/slope for each
        n_raw = n_feat
        combined_imp = np.zeros(n_raw)
        for i in range(n_raw):
            combined_imp[i] = (rf.feature_importances_[i]
                               + rf.feature_importances_[i + n_raw]
                               + rf.feature_importances_[i + 2 * n_raw])
        ranked = sorted(zip(feature_cols, combined_imp), key=lambda x: -x[1])
        _print_ranking("RF IMPURITY FEATURE RANKING", ranked)

    # ── Permutation importance (RF-based, model-agnostic) ──
    # Evaluated on training data (not early-stop) to avoid double-dipping:
    # early-stop already selects checkpoints, using it here too would create
    # a mild overfitting loop (features chosen to look good on early-stop,
    # then checkpoint also chosen on early-stop with those features).
    if CONFIG.get("permutation_importance"):
        X_train_rf = build_rf_features(X_train)
        y_train_mean = y_train.mean(axis=1)
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=42)
        rf.fit(X_train_rf, y_train_mean)
        result = sklearn_perm_importance(rf, X_train_rf, y_train_mean,
                                         n_repeats=5, random_state=42, n_jobs=-1)
        # Map back to raw features
        n_raw = n_feat
        combined_imp = np.zeros(n_raw)
        for i in range(n_raw):
            combined_imp[i] = (result.importances_mean[i]
                               + result.importances_mean[i + n_raw]
                               + result.importances_mean[i + 2 * n_raw])
        ranked = sorted(zip(feature_cols, combined_imp), key=lambda x: -x[1])
        _print_ranking("PERMUTATION IMPORTANCE FEATURE RANKING", ranked)

    # ── Gradient-based importance (LSTM) ──
    if CONFIG.get("gradient_feature_selection"):
        seeds = [42, 123, 456]
        importance_accum = np.zeros(n_feat, dtype=np.float64)
        n_steps = CONFIG["horizon"] // CONFIG["sub_window"]

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = PlainLSTM(input_size=n_feat, n_steps=n_steps).to(device)
            optimizer = Adam(model.parameters(), lr=CONFIG["lr"],
                             weight_decay=CONFIG["weight_decay"])
            mse_loss = nn.MSELoss()
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                batch_size=CONFIG["batch_size"], shuffle=True)

            feat_grad_sum = np.zeros(n_feat, dtype=np.float64)
            n_batches = 0

            for epoch in range(min(50, CONFIG["max_epochs"])):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    xb.requires_grad_(True)
                    loss = mse_loss(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        grad_mag = xb.grad.abs().mean(dim=(0, 1))
                        feat_grad_sum += grad_mag.cpu().numpy()
                        n_batches += 1
                    nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                    optimizer.step()

            importance_accum += feat_grad_sum / max(n_batches, 1)
            print(f"  Gradient importance: seed {seed} done ({n_batches} batches)",
                  file=sys.stderr)

        importance_accum /= len(seeds)
        ranked = sorted(zip(feature_cols, importance_accum), key=lambda x: -x[1])
        _print_ranking("GRADIENT FEATURE RANKING", ranked)


def _sffs_eval_rf(feature_subset, train_df, within_val_df):
    """Train RF on a feature subset, return MSE on within-val set.

    Uses within-val (not early-stop) to avoid double-dipping: SFFS makes
    hundreds of evaluations iteratively optimizing features against this set.
    If early-stop were used, the features would be tuned to it, then checkpoint
    selection would also use early-stop — creating a leakage loop.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_subset].values)

    X_train, y_train, _, _ = build_windows(
        train_df, feature_subset, CONFIG["train_stride"], scaler)
    X_val, y_val, _, _ = build_windows(
        within_val_df, feature_subset, CONFIG["eval_stride"], scaler)

    if len(y_val) == 0 or len(y_train) == 0:
        return float("inf")

    X_train_rf = build_rf_features(X_train)
    X_val_rf = build_rf_features(X_val)
    y_train_mean = y_train.mean(axis=1)
    y_val_mean = y_val.mean(axis=1)

    rf = RandomForestRegressor(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        min_samples_leaf=CONFIG["rf_min_samples_leaf"],
        max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train_mean)

    pred = rf.predict(X_val_rf)
    return float(mean_squared_error(y_val_mean, pred))


def run_sffs(all_features, train_df, within_val_df):
    """Sequential Forward Floating Selection (SFFS).

    Greedy forward selection with backtracking: after each feature addition,
    try removing each existing feature — if removal improves MSE, drop it.
    This escapes local optima that plain forward selection gets stuck in.

    Prints progress and the best subset at each size. Outputs a CONFIG-ready
    feature list to stdout.
    """
    max_k = CONFIG.get("sffs_max_features", 50)
    start_features = CONFIG.get("sffs_start_features") or []

    selected = list(start_features)
    remaining = [f for f in all_features if f not in selected]

    # Evaluate starting point
    if selected:
        best_mse = _sffs_eval_rf(selected, train_df, within_val_df)
        print(f"  SFFS start: {len(selected)} features, mse={best_mse:.6f}",
              file=sys.stderr)
    else:
        best_mse = float("inf")

    # Track best subset at each size for the final report
    best_at_k = {}
    if selected:
        best_at_k[len(selected)] = (best_mse, list(selected))

    t0 = time.time()
    n_evals = 0

    while len(selected) < max_k and remaining:
        # ── Forward step: find best feature to add ──
        best_add_mse = float("inf")
        best_add_feat = None

        for feat in remaining:
            candidate = selected + [feat]
            mse = _sffs_eval_rf(candidate, train_df, within_val_df)
            n_evals += 1
            if mse < best_add_mse:
                best_add_mse = mse
                best_add_feat = feat

        selected.append(best_add_feat)
        remaining.remove(best_add_feat)
        best_mse = best_add_mse
        k = len(selected)

        elapsed = time.time() - t0
        print(f"  SFFS +{best_add_feat:<40s} k={k:2d}  "
              f"mse={best_mse:.6f}  ({n_evals} evals, {elapsed:.0f}s)",
              file=sys.stderr)

        # ── Backward step: try removing each feature ──
        improved = True
        while improved and len(selected) > 1:
            improved = False
            best_drop_mse = best_mse
            best_drop_feat = None

            for feat in selected:
                candidate = [f for f in selected if f != feat]
                mse = _sffs_eval_rf(candidate, train_df, within_val_df)
                n_evals += 1
                if mse < best_drop_mse:
                    best_drop_mse = mse
                    best_drop_feat = feat

            if best_drop_feat is not None:
                selected.remove(best_drop_feat)
                remaining.append(best_drop_feat)
                best_mse = best_drop_mse
                k = len(selected)
                improved = True
                print(f"  SFFS -{best_drop_feat:<40s} k={k:2d}  "
                      f"mse={best_mse:.6f}  (backtrack)", file=sys.stderr)

        # Record best at this size
        k = len(selected)
        if k not in best_at_k or best_mse < best_at_k[k][0]:
            best_at_k[k] = (best_mse, list(selected))

    # ── Report ──
    elapsed = time.time() - t0
    print(f"\n  SFFS complete: {n_evals} RF evaluations in {elapsed:.0f}s",
          file=sys.stderr)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("SFFS RESULTS (best MSE at each subset size)", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  {'k':>3}  {'MSE':>12}", file=sys.stderr)
    print(f"  {'-' * 17}", file=sys.stderr)

    overall_best_k = min(best_at_k, key=lambda k: best_at_k[k][0])
    for k in sorted(best_at_k):
        mse, _ = best_at_k[k]
        marker = " <-- BEST" if k == overall_best_k else ""
        print(f"  {k:>3}  {mse:>12.6f}{marker}", file=sys.stderr)

    best_mse_final, best_subset_final = best_at_k[overall_best_k]

    # Print CONFIG-ready output to stdout
    print(f"\n# SFFS best subset: {overall_best_k} features, "
          f"mse={best_mse_final:.6f}")
    print("SFFS_BEST_FEATURES = [")
    for f in best_subset_final:
        print(f'    "{f}",')
    print("]")

    return best_subset_final


# ═══════════════════════════════════════════════════════════════════
# RF FEATURES (flatten windows to summary stats)
# ═══════════════════════════════════════════════════════════════════

def build_rf_features(X_windows):
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()

    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64), t_centered) / t_var
    return np.hstack([means, stds, slopes.astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# PLAIN LSTM (MSE loss, single scalar output per step)
# ═══════════════════════════════════════════════════════════════════

class PlainLSTM(nn.Module):
    def __init__(self, input_size, n_steps):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0
        self.n_steps = n_steps
        self.hidden = hidden

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTMCell(input_size=1, hidden_size=hidden)
        self.step_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        # Use top-layer hidden/cell as decoder initial state
        h_dec = h_n[-1]  # (batch, hidden)
        c_dec = c_n[-1]

        preds = []
        # First decoder input: zero (no previous prediction)
        dec_input = torch.zeros(x.size(0), 1, device=x.device)
        for _ in range(self.n_steps):
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            step_pred = self.step_head(h_dec)  # (batch, 1)
            preds.append(step_pred)
            dec_input = step_pred.detach()  # feed prediction as next input

        return torch.cat(preds, dim=1)  # (batch, n_steps)


# ═══════════════════════════════════════════════════════════════════
# EVIDENTIAL LSTM (NIG output for uncertainty quantification)
# ═══════════════════════════════════════════════════════════════════

class EvidentialLSTM(nn.Module):
    """LSTM with Normal-Inverse-Gamma output for evidential regression.
    Autoregressive decoder outputs NIG params (γ, ν, α, β) per step."""

    def __init__(self, input_size, n_steps):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0
        self.n_steps = n_steps
        self.hidden = hidden

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTMCell(input_size=1, hidden_size=hidden)
        self.step_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, 4),  # 4 NIG params per step
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_dec = h_n[-1]
        c_dec = c_n[-1]

        gammas, nus, alphas, betas = [], [], [], []
        dec_input = torch.zeros(x.size(0), 1, device=x.device)
        for _ in range(self.n_steps):
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            raw = self.step_head(h_dec)  # (batch, 4)

            gamma = raw[:, 0]
            nu = self.softplus(raw[:, 1]) + 1e-6
            alpha = self.softplus(raw[:, 2]) + 1.0 + 1e-6
            beta = self.softplus(raw[:, 3]) + 1e-6

            gammas.append(gamma)
            nus.append(nu)
            alphas.append(alpha)
            betas.append(beta)

            dec_input = gamma.unsqueeze(1).detach()  # feed γ as next input

        return (torch.stack(gammas, dim=1), torch.stack(nus, dim=1),
                torch.stack(alphas, dim=1), torch.stack(betas, dim=1))

    def predict(self, x):
        gamma, nu, alpha, beta = self.forward(x)
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return gamma, aleatoric, epistemic, nu, alpha, beta


# ═══════════════════════════════════════════════════════════════════
# NIG LOSS (Amini et al. 2020)
# ═══════════════════════════════════════════════════════════════════

def nig_loss(gamma, nu, alpha, beta, y):
    """NIG evidential regression loss, averaged over steps."""
    omega = 2.0 * beta * (1.0 + nu)
    nll = (0.5 * torch.log(np.pi / nu)
           - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))

    error = torch.abs(y - gamma)
    # NOTE: Amini et al. (2020, Eq. 9) define evidence as (2ν + α). We use
    # (2ν + α − 1) to exclude the +1.0 floor baked into α's parameterization,
    # so that zero learned evidence → zero regularization penalty. Without this,
    # the regularizer introduces a hidden MAE term even at the prior (α=1, ν=0).
    # See Meinert & Lavin (2022) for discussion of regularizer pathologies.
    evidence = 2.0 * nu + (alpha - 1.0)
    reg_evidence = error * evidence

    loss = nll + CONFIG["lambda1"] * reg_evidence
    return loss.mean()


# ═══════════════════════════════════════════════════════════════════
# STUDENT-T CDF → 3-CLASS PROBABILITIES
# ═══════════════════════════════════════════════════════════════════

def nig_to_class_probs(gamma, nu, alpha, beta, t1, t2):
    """Convert NIG parameters to P(easy), P(moderate), P(hard) via Student-t CDF."""
    if isinstance(gamma, torch.Tensor):
        gamma = gamma.cpu().numpy()
        nu = nu.cpu().numpy()
        alpha = alpha.cpu().numpy()
        beta = beta.cpu().numpy()

    df = 2.0 * alpha
    loc = gamma
    scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))

    p_easy = np.zeros(len(gamma))
    p_moderate = np.zeros(len(gamma))
    p_hard = np.zeros(len(gamma))

    for i in range(len(gamma)):
        if df[i] <= 0 or scale[i] <= 0 or np.isnan(df[i]) or np.isnan(scale[i]):
            p_easy[i] = 1.0 / 3.0
            p_moderate[i] = 1.0 / 3.0
            p_hard[i] = 1.0 / 3.0
            continue

        dist = scipy_stats.t(df=df[i], loc=loc[i], scale=scale[i])
        cdf_t1 = dist.cdf(t1)
        cdf_t2 = dist.cdf(t2)

        p_easy[i] = np.clip(cdf_t1, 1e-8, 1.0 - 1e-8)
        p_hard[i] = np.clip(1.0 - cdf_t2, 1e-8, 1.0 - 1e-8)
        p_moderate[i] = np.clip(cdf_t2 - cdf_t1, 1e-8, 1.0 - 1e-8)

        total = p_easy[i] + p_moderate[i] + p_hard[i]
        p_easy[i] /= total
        p_moderate[i] /= total
        p_hard[i] /= total

    return p_easy, p_moderate, p_hard


# ═══════════════════════════════════════════════════════════════════
# TRAINING — PLAIN LSTM
# ═══════════════════════════════════════════════════════════════════

def train_plain(X_train, y_train, X_val, y_val, n_feat, n_steps, device):
    model = PlainLSTM(input_size=n_feat, n_steps=n_steps).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"],
                     weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse_loss = nn.MSELoss()

    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = mse_loss(model(xb), yb)
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = mse_loss(model(val_x), val_y).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= CONFIG["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# TRAINING — EVIDENTIAL LSTM
# ═══════════════════════════════════════════════════════════════════

def train_evidential(X_train, y_train, X_val, y_val, n_feat, n_steps, device):
    model = EvidentialLSTM(input_size=n_feat, n_steps=n_steps).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"],
                     weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["max_epochs"], eta_min=1e-6)

    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            gamma, nu, alpha, beta = model(xb)
            loss = nig_loss(gamma, nu, alpha, beta, yb)

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at epoch {epoch}", file=sys.stderr)
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

        model.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = model(val_x)
            val_loss = nig_loss(gamma, nu, alpha, beta, val_y).item()

        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if not np.isnan(val_loss):
            scheduler.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: val_loss={val_loss:.4f}"
                  + (" *" if patience_ctr == 0 else ""), file=sys.stderr)

        if patience_ctr >= CONFIG["patience"]:
            print(f"  Early stop at epoch {epoch}", file=sys.stderr)
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def transition_accuracy_3class(y_true, y_pred, boundary_indices=None):
    """Accuracy on windows where the true 3-class label changes.

    Skips comparisons at video boundary indices where consecutive windows
    come from different videos and are not temporally related.
    """
    if boundary_indices is None:
        boundary_indices = set()
    correct, total = 0, 0
    for j in range(1, len(y_true)):
        if j in boundary_indices:
            continue
        if y_true[j] != y_true[j - 1]:
            total += 1
            if y_pred[j] == y_true[j]:
                correct += 1
    return (correct / total if total > 0 else 0.0), total


def to_3class(y_continuous, t1, t2):
    """Convert continuous miss_rate to 3 classes: 0=easy, 1=moderate, 2=hard."""
    classes = np.zeros(len(y_continuous), dtype=np.int64)
    classes[(y_continuous >= t1) & (y_continuous < t2)] = 1
    classes[y_continuous >= t2] = 2
    return classes


def eval_regression(y_true_mean, y_pred_mean, t1, t2, boundary_indices=None):
    """Evaluate regression predictions (plain mode or evidential gamma)."""
    mse = float(mean_squared_error(y_true_mean, y_pred_mean))
    mae = float(mean_absolute_error(y_true_mean, y_pred_mean))

    true_cls = to_3class(y_true_mean, t1, t2)
    pred_cls = to_3class(y_pred_mean, t1, t2)
    majority_acc = float(np.max(np.bincount(true_cls)) / len(true_cls))
    cls_acc = float(np.mean(pred_cls == true_cls))
    cls_trans, n_trans = transition_accuracy_3class(true_cls, pred_cls, boundary_indices)

    return {"mse": mse, "mae": mae, "cls_acc": cls_acc,
            "cls_trans_acc": cls_trans, "n_trans": n_trans,
            "majority_acc": majority_acc}


def eval_evidential(model, X, y, device, t1, t2, boundary_indices=None):
    """Full evidential evaluation: regression + uncertainty + 3-class via Student-t CDF."""
    val_x = torch.from_numpy(X).to(device)
    n_steps = y.shape[1]

    with torch.no_grad():
        gamma, aleatoric, epistemic, nu, alpha, beta = model.predict(val_x)

    gamma_mean = gamma.cpu().numpy().mean(axis=1)
    y_mean = y.mean(axis=1)
    epistemic_mean = epistemic.cpu().numpy().mean(axis=1)

    mse = float(np.mean((gamma_mean - y_mean) ** 2))
    mae = float(np.mean(np.abs(gamma_mean - y_mean)))

    # 3-class via Student-t CDF: average probabilities across steps
    p_easy_steps, p_mod_steps, p_hard_steps = [], [], []
    for s in range(n_steps):
        pe, pm, ph = nig_to_class_probs(
            gamma[:, s], nu[:, s], alpha[:, s], beta[:, s], t1, t2)
        p_easy_steps.append(pe)
        p_mod_steps.append(pm)
        p_hard_steps.append(ph)
    p_easy = np.mean(p_easy_steps, axis=0)
    p_moderate = np.mean(p_mod_steps, axis=0)
    p_hard = np.mean(p_hard_steps, axis=0)

    true_cls = to_3class(y_mean, t1, t2)
    majority_acc = float(np.max(np.bincount(true_cls)) / len(true_cls))

    # Threshold-based classification (comparable to LSTM/RF)
    pred_cls_thresh = to_3class(gamma_mean, t1, t2)
    cls_acc_thresh = float(np.mean(pred_cls_thresh == true_cls))
    cls_trans_thresh, n_trans = transition_accuracy_3class(
        true_cls, pred_cls_thresh, boundary_indices)

    # CDF-based classification (evidential-specific, uses uncertainty)
    probs = np.stack([p_easy, p_moderate, p_hard], axis=1)
    pred_cls_cdf = probs.argmax(axis=1)
    cls_acc_cdf = float(np.mean(pred_cls_cdf == true_cls))
    cls_trans_cdf, _ = transition_accuracy_3class(
        true_cls, pred_cls_cdf, boundary_indices)

    # Uncertainty calibration (use threshold-based for consistency)
    correct_mask = (pred_cls_thresh == true_cls)
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        unc_separation = float(epistemic_mean[~correct_mask].mean()
                               - epistemic_mean[correct_mask].mean())
    else:
        unc_separation = 0.0

    return {"mse": mse, "mae": mae,
            "cls_acc": cls_acc_thresh, "cls_trans_acc": cls_trans_thresh,
            "cls_acc_cdf": cls_acc_cdf, "cls_trans_acc_cdf": cls_trans_cdf,
            "n_trans": n_trans, "unc_separation": unc_separation,
            "majority_acc": majority_acc}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Argparse ──
    parser = argparse.ArgumentParser(description="ECHO Bellevue two-sweep training")
    parser.add_argument("--mode", choices=["lstm", "evidential"], default=None,
                        help="Override CONFIG mode: lstm or evidential")
    parser.add_argument("--results-file", default=None,
                        help="Results TSV file (default: results/pod_results_evid.tsv)")
    parser.add_argument("--stability", action="store_true",
                        help="Run 3-seed stability test and exit")
    args = parser.parse_args()

    if args.mode:
        CONFIG["mode"] = args.mode
    mode = CONFIG["mode"]
    assert mode in ("lstm", "evidential"), f"Invalid mode: {mode}"

    results_file = args.results_file
    if results_file is None:
        fname = "pod_results_lstm.tsv" if mode == "lstm" else "pod_results_evid.tsv"
        results_file = str(RESULTS_DIR / fname)

    t0 = time.time()
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU!", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Mode: {mode}", file=sys.stderr)
    print(f"  Results file: {results_file}", file=sys.stderr)

    df = load_data()
    feature_cols = CONFIG["features"]
    target_col = get_target_column()

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}", file=sys.stderr)
        sys.exit(1)

    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S

    # Keep all frames including x_count=0: filtering would create temporal
    # gaps that the LSTM sees as contiguous, distorting windowed sequences.
    # miss_rate for x_count=0 frames is 0/1=0 (no vehicles to miss).
    n_zero = int((df["x_count"] == 0).sum())
    if n_zero:
        print(f"  x_count=0 frames kept: {n_zero}/{len(df)} "
              f"({100*n_zero/len(df):.1f}%)", file=sys.stderr)

    # Split (3-way from training intersections + held-out cross-camera)
    train_df, earlystop_df, within_val_df, cross_val_df = split_data(df)
    print(f"  Multi-step: horizon={H}, sub_window={S}, n_steps={n_steps}", file=sys.stderr)
    print(f"  Train: {len(train_df)} frames from {CONFIG['train_intersections']}",
          file=sys.stderr)
    print(f"  Early-stop: {len(earlystop_df)} frames", file=sys.stderr)
    print(f"  Within-val: {len(within_val_df)} frames", file=sys.stderr)
    print(f"  Cross-val: {len(cross_val_df)} frames from {CONFIG['test_intersection']}",
          file=sys.stderr)

    # ── SFFS (runs before normal windowing, uses ALL_FEATURES) ──
    if CONFIG.get("sffs_feature_selection"):
        print("\n  ── Sequential Forward Floating Selection ──", file=sys.stderr)
        sffs_features = ALL_FEATURES
        sffs_missing = [f for f in sffs_features if f not in df.columns]
        if sffs_missing:
            print(f"ERROR: Missing features for SFFS: {sffs_missing}", file=sys.stderr)
            sys.exit(1)
        run_sffs(sffs_features, train_df, within_val_df)
        print("\n  SFFS complete. Update CONFIG['features'] with the result and re-run.",
              file=sys.stderr)
        sys.exit(0)

    # Scaler from training data only (not early-stop or val)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # Build windows for all 4 sets
    X_train, y_train, _, _ = build_windows(
        train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_earlystop, y_earlystop, _, _ = build_windows(
        earlystop_df, feature_cols, CONFIG["eval_stride"], scaler)
    X_within, y_within, bnd_within, seq_labels_within = build_windows(
        within_val_df, feature_cols, CONFIG["eval_stride"], scaler)
    X_cross, y_cross, bnd_cross, _ = build_windows(
        cross_val_df, feature_cols, CONFIG["eval_stride"], scaler)

    n_feat = len(feature_cols)
    y_within_mean = y_within.mean(axis=1)
    y_cross_mean = y_cross.mean(axis=1)

    print(f"  Windows: train={len(y_train)}, earlystop={len(y_earlystop)}, "
          f"within={len(y_within)}, cross={len(y_cross)}", file=sys.stderr)

    if len(y_train) == 0 or len(y_earlystop) == 0 or len(y_within) == 0 or len(y_cross) == 0:
        print(f"ERROR: Empty split (train={len(y_train)}, earlystop={len(y_earlystop)}, "
              f"within={len(y_within)}, cross={len(y_cross)}). "
              f"Check data, window/horizon settings, or warmup_frames.",
              file=sys.stderr)
        sys.exit(1)

    # ── Feature selection (if any flags are enabled) ──
    any_fs = any(CONFIG.get(k) for k in [
        "spearman_feature_selection", "mi_feature_selection",
        "lasso_feature_selection", "elasticnet_feature_selection",
        "rfe_feature_selection", "rf_feature_importance",
        "permutation_importance", "gradient_feature_selection"])
    if any_fs:
        print("\n  ── Feature Selection ──", file=sys.stderr)
        run_feature_selection(train_df, feature_cols, X_train, y_train,
                              n_feat, device)

    # Difficulty thresholds from training data
    t1 = float(np.percentile(y_train.flatten(), CONFIG["t1_percentile"]))
    t2 = float(np.percentile(y_train.flatten(), CONFIG["t2_percentile"]))
    print(f"  Thresholds: t1={t1:.4f} (p{CONFIG['t1_percentile']}), "
          f"t2={t2:.4f} (p{CONFIG['t2_percentile']})", file=sys.stderr)

    # ── Stability test (evidential only) ──
    if args.stability:
        stability_seeds = [42, 123, 456]
        stability_mses = {}
        for seed in stability_seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = train_evidential(X_train, y_train, X_earlystop, y_earlystop,
                                     n_feat, n_steps, device)
            m = eval_evidential(model, X_within, y_within, device, t1, t2, bnd_within)
            stability_mses[seed] = m["mse"]
        mse_vals = list(stability_mses.values())
        mse_range = max(mse_vals) - min(mse_vals)
        parts = " ".join(f"seed {s} mse={v:.6f}," for s, v in stability_mses.items())
        print(f"STABILITY: {parts} range={mse_range:.6f}", file=sys.stderr)
        sys.exit(0)

    # ═════════════════════════════════════════════════════════════
    # TRAIN SINGLE MODEL BASED ON MODE
    # ═════════════════════════════════════════════════════════════

    if mode == "lstm":
        # ── PlainLSTM (MSE loss, 3-seed ensemble) ──
        print("\n  ── LSTM (3-seed ensemble) ──", file=sys.stderr)
        all_preds_within, all_preds_cross = [], []
        for seed in CONFIG["seeds"]:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = train_plain(X_train, y_train, X_earlystop, y_earlystop,
                                n_feat, n_steps, device)
            with torch.no_grad():
                pw = model(torch.from_numpy(X_within).to(device)).cpu().numpy()
                pc = model(torch.from_numpy(X_cross).to(device)).cpu().numpy()
            all_preds_within.append(pw)
            all_preds_cross.append(pc)

        pred_within = np.mean(all_preds_within, axis=0).mean(axis=1)
        pred_cross = np.mean(all_preds_cross, axis=0).mean(axis=1)

        m_within = eval_regression(y_within_mean, pred_within, t1, t2, bnd_within)
        m_cross = eval_regression(y_cross_mean, pred_cross, t1, t2, bnd_cross)

        model_name = "LSTM"
        print(f"  LSTM within: mse={m_within['mse']:.4f} "
              f"cls_acc={m_within['cls_acc']:.3f} "
              f"cls_trans={m_within['cls_trans_acc']:.3f}", file=sys.stderr)
        print(f"  LSTM cross:  mse={m_cross['mse']:.4f} "
              f"cls_acc={m_cross['cls_acc']:.3f} "
              f"cls_trans={m_cross['cls_trans_acc']:.3f}", file=sys.stderr)

    else:
        # ── EvidentialLSTM (NIG loss, single seed) ──
        print("\n  ── EvidentialLSTM (single seed) ──", file=sys.stderr)
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        evid_model = train_evidential(X_train, y_train, X_earlystop, y_earlystop,
                                      n_feat, n_steps, device)

        m_within = eval_evidential(evid_model, X_within, y_within, device, t1, t2, bnd_within)
        m_cross = eval_evidential(evid_model, X_cross, y_cross, device, t1, t2, bnd_cross)

        model_name = "EvidentialLSTM"
        print(f"  EVID within: mse={m_within['mse']:.4f} "
              f"cls_acc={m_within['cls_acc']:.3f} "
              f"cls_acc_cdf={m_within['cls_acc_cdf']:.3f} "
              f"unc_sep={m_within['unc_separation']:.4f}", file=sys.stderr)
        print(f"  EVID cross:  mse={m_cross['mse']:.4f} "
              f"cls_acc={m_cross['cls_acc']:.3f} "
              f"cls_acc_cdf={m_cross['cls_acc_cdf']:.3f} "
              f"unc_sep={m_cross['unc_separation']:.4f}", file=sys.stderr)

    # ── Per-intersection within-MSE breakdown ──
    seq_arr = np.array(seq_labels_within)
    per_int_parts = []
    for int_name in CONFIG["train_intersections"]:
        mask = seq_arr == int_name
        if mask.sum() > 0:
            if mode == "lstm":
                int_mse = float(mean_squared_error(y_within_mean[mask], pred_within[mask]))
            else:
                gamma_within = evid_model.predict(
                    torch.from_numpy(X_within[mask]).to(device))[0].cpu().numpy().mean(axis=1)
                int_mse = float(mean_squared_error(y_within_mean[mask], gamma_within))
            short_name = int_name.split("_")[-1]
            per_int_parts.append(f"{short_name}={int_mse:.3f}")
    if per_int_parts:
        print(f"  Per-intersection within-MSE: {' '.join(per_int_parts)}", file=sys.stderr)

    # ── Cross/within MSE ratio diagnostic ──
    print(f"  DIAGNOSTIC: cross_mse/within_mse = {m_cross['mse']/m_within['mse']:.2f}",
          file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n  {model_name}  Time: {elapsed:.1f}s", file=sys.stderr)

    # ═════════════════════════════════════════════════════════════
    # RESULT LINE
    # ═════════════════════════════════════════════════════════════

    config_summary = {
        "mode": mode,
        "model": model_name,
        "n_features": n_feat,
        "window": CONFIG["window"],
        "horizon": CONFIG["horizon"],
        "sub_window": CONFIG["sub_window"],
        "n_steps": n_steps,
        "hidden_size": CONFIG["hidden_size"],
        "n_layers": CONFIG["n_layers"],
        "dropout": CONFIG["dropout"],
        "lr": CONFIG["lr"],
        "batch_size": CONFIG["batch_size"],
        "t1_percentile": CONFIG["t1_percentile"],
        "t2_percentile": CONFIG["t2_percentile"],
        "t1": round(t1, 4),
        "t2": round(t2, 4),
    }
    if mode == "evidential":
        config_summary["lambda1"] = CONFIG["lambda1"]

    result_fields = [
        "RESULT",
        f"model={model_name}",
        f"mse_within={m_within['mse']:.6f}",
        f"mse_cross={m_cross['mse']:.6f}",
        f"mae_within={m_within['mae']:.6f}",
        f"mae_cross={m_cross['mae']:.6f}",
        f"cls_acc_within={m_within['cls_acc']:.6f}",
        f"cls_acc_cross={m_cross['cls_acc']:.6f}",
        f"cls_trans_within={m_within['cls_trans_acc']:.6f}",
        f"cls_trans_cross={m_cross['cls_trans_acc']:.6f}",
        f"majority_acc_within={m_within['majority_acc']:.6f}",
        f"majority_acc_cross={m_cross['majority_acc']:.6f}",
    ]

    if mode == "evidential":
        result_fields.extend([
            f"unc_sep_within={m_within['unc_separation']:.6f}",
            f"unc_sep_cross={m_cross['unc_separation']:.6f}",
            f"cls_acc_cdf_within={m_within['cls_acc_cdf']:.6f}",
            f"cls_acc_cdf_cross={m_cross['cls_acc_cdf']:.6f}",
            f"cls_trans_cdf_within={m_within['cls_trans_acc_cdf']:.6f}",
            f"cls_trans_cdf_cross={m_cross['cls_trans_acc_cdf']:.6f}",
        ])

    result_fields.extend([
        f"n_within={len(y_within)}",
        f"n_cross={len(y_cross)}",
        f"config={json.dumps(config_summary)}",
    ])

    print("\t".join(result_fields))


if __name__ == "__main__":
    main()
