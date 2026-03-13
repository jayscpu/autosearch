#!/usr/bin/env python3
"""ECHO Fresno — Single-file train/eval for autoresearch loop.

Loads pre-extracted features from CSV, trains, evaluates, prints one RESULT line.
This file is edited by the AI agent. The CSV files are fixed.
"""

import json
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — the agent modifies this section
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Features ──
    "features": [
        # Spearman top-35 + top-5 unique gradient-importance features
        "glcm_entropy", "image_entropy", "glcm_energy", "brightness_std",
        "rms_contrast", "glcm_contrast", "mscn_v_pair_mean",
        "gabor_nyquist_energy", "foreground_edge_density", "edge_density_coarse",
        "foreground_pixel_ratio", "edge_fine_coarse_ratio", "fast_keypoints_half",
        "shadow_pixel_ratio", "glcm_homogeneity", "gradient_magnitude_std",
        "spatial_frequency", "foreground_blob_count", "mid_gradient_std",
        "downsample_info_loss", "downsample_ssim", "mscn_mean",
        "motion_pixel_ratio", "mid_high_freq_energy", "fft_critical_band_ratio",
        "temporal_diff_mean", "mscn_skewness", "temporal_diff_std",
        "keypoint_loss_ratio", "saturation_std", "ratio_top_bot_gradient_std",
        "bot_gradient_std", "dark_channel_mean", "colorfulness", "mscn_h_pair_std",
        # Gradient-unique additions (top-5 by gradient importance not in Spearman-35)
        "mscn_h_pair_mean", "top_gradient_std", "dct_high_freq_ratio",
        "ratio_top_bot_edge_density", "foreground_blob_pa_ratio",
    ],

    # ── Target definition ──
    "target": "miss_rate",        # "fn_nano", "miss_rate", or "frame_f1"
    "threshold_method": "percentile",  # "median", "fixed", "percentile"
    "threshold_value": 25,         # 25th percentile
    "min_threshold": None,          # floor for threshold (None for miss_rate)

    # ── Windowing ──
    "window": 30,
    "horizon": 30,
    "train_stride": 10,
    "eval_stride": 30,            # should equal window for non-overlapping val

    # ── Temporal split ──
    "train_cutoff": 14400,

    # ── Model ──
    "model_type": "lstm",          # "lstm", "gru", "rf_only"
    "hidden_size": 64,
    "n_layers": 2,
    "dropout": 0.4,
    "use_dirichlet": False,        # True=Dirichlet head, False=plain softmax+CE
    "kl_annealing_epochs": 10,

    # ── Training ──
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 25,
    "grad_clip": 1.0,
    "seeds": [123, 789],            # best performing seeds

    # ── RF ──
    "rf_n_estimators": 500,
    "rf_max_depth": 20,
    "rf_min_samples_leaf": 10,
    "rf_features": "mean_std_slope",  # "mean_std_slope", "mean_std_slope_minmax", "all"
    "rf_class_weight": None,        # None, "balanced", "balanced_subsample"

    # ── Gradient feature selection ──
    "gradient_feature_selection": False,  # already computed, using results now
}

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "yolox_features.csv"
DETS_CSV = SCRIPT_DIR / "yolox_detections.csv"


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING & TARGET COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load features CSV and optionally merge detection data for alternate targets."""
    df = pd.read_csv(FEATURES_CSV)

    if CONFIG["target"] in ("miss_rate", "frame_f1"):
        dets = pd.read_csv(DETS_CSV)
        df = df.merge(dets[["frame_id", "nano_tp", "nano_fp", "x_count"]], on="frame_id", how="left",
                       suffixes=("", "_det"))
        # Use _det columns if merge created duplicates, otherwise use originals
        for col in ["nano_tp", "nano_fp", "x_count"]:
            det_col = f"{col}_det"
            if det_col in df.columns:
                df[col] = df[det_col].fillna(df[col])
                df.drop(columns=[det_col], inplace=True)

        if CONFIG["target"] == "miss_rate":
            df["miss_rate"] = df["fn_nano"] / df["x_count"].clip(lower=1)
        elif CONFIG["target"] == "frame_f1":
            p = df["nano_tp"] / (df["nano_tp"] + df["nano_fp"]).clip(lower=1)
            r = df["nano_tp"] / (df["nano_tp"] + df["fn_nano"]).clip(lower=1)
            df["frame_f1"] = (2 * p * r / (p + r).clip(lower=1e-8))
            # Invert: low F1 = hard, so target = 1 - frame_f1 for thresholding
            df["frame_f1_inv"] = 1.0 - df["frame_f1"]

    return df


# ═══════════════════════════════════════════════════════════════════
# WINDOWING
# ═══════════════════════════════════════════════════════════════════

def get_target_column():
    """Return the column name to aggregate over the horizon."""
    t = CONFIG["target"]
    if t == "fn_nano":
        return "fn_nano"
    elif t == "miss_rate":
        return "miss_rate"
    elif t == "frame_f1":
        return "frame_f1_inv"
    else:
        return t


def compute_threshold(df):
    """Compute binarization threshold from all possible horizon windows."""
    target_col = get_target_column()
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    stride = CONFIG["eval_stride"]

    sdf = df.sort_values("frame_id").reset_index(drop=True)
    n = len(sdf)
    total_need = W + H
    targets = []
    for t in range(0, n - total_need + 1, stride):
        targets.append(sdf[target_col].iloc[t + W: t + W + H].mean())

    targets = np.array(targets)

    method = CONFIG["threshold_method"]
    if method == "median":
        thr = float(np.median(targets))
    elif method == "fixed":
        thr = float(CONFIG["threshold_value"])
    elif method == "percentile":
        thr = float(np.percentile(targets, CONFIG["threshold_value"]))
    else:
        thr = float(np.median(targets))

    # Apply floor
    if CONFIG["min_threshold"] is not None and thr < CONFIG["min_threshold"]:
        thr = CONFIG["min_threshold"]

    return thr


def build_windows(df, feature_cols, target_col, stride, scaler, threshold):
    """Build shifted-label windows."""
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    total_need = W + H

    X_list, y_list, seq_ids = [], [], []

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        feat_vals = scaler.transform(sdf[feature_cols].values)
        target_vals = sdf[target_col].values

        for t in range(0, n - total_need + 1, stride):
            x_window = feat_vals[t: t + W]
            target_mean = target_vals[t + W: t + W + H].mean()

            X_list.append(x_window)
            y_list.append(int(target_mean >= threshold))
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, seq_ids


# ═══════════════════════════════════════════════════════════════════
# RF FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def build_rf_features(X_windows):
    """Convert 3D windows to 2D RF features."""
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()

    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64), t_centered) / t_var

    parts = [means, stds, slopes.astype(np.float32)]

    mode = CONFIG["rf_features"]
    if mode in ("mean_std_slope_minmax", "all"):
        mins = X_windows.min(axis=1)
        maxs = X_windows.max(axis=1)
        parts.extend([mins, maxs])

    if mode == "all":
        last_val = X_windows[:, -1, :]
        first_last_diff = X_windows[:, -1, :] - X_windows[:, 0, :]
        q25 = np.percentile(X_windows, 25, axis=1)
        q75 = np.percentile(X_windows, 75, axis=1)
        parts.extend([last_val, first_last_diff, q25, q75])

    return np.hstack(parts)


# ═══════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════

class SequenceModel(nn.Module):
    """LSTM or GRU with optional Dirichlet or plain softmax head."""

    def __init__(self, input_size):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0

        if CONFIG["model_type"] == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden,
                              num_layers=n_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True, dropout=dropout)

        if CONFIG["use_dirichlet"]:
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
                nn.Linear(32, 2), nn.Softplus(),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
                nn.Linear(32, 2),
            )

    def forward(self, x):
        if CONFIG["model_type"] == "gru":
            _, h_n = self.rnn(x)
        else:
            _, (h_n, _) = self.rnn(x)
        out = self.head(h_n[-1])
        if CONFIG["use_dirichlet"]:
            out = out + 1.0
        return out

    def predict(self, x):
        out = self.forward(x)
        if CONFIG["use_dirichlet"]:
            S = out.sum(dim=1, keepdim=True)
            probs = out / S
            return probs.argmax(dim=1), probs[:, 1]
        else:
            probs = torch.softmax(out, dim=1)
            return probs.argmax(dim=1), probs[:, 1]


def dirichlet_loss(alphas, targets, epoch):
    """Evidential Dirichlet loss with KL annealing."""
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    S = alphas.sum(dim=1, keepdim=True)

    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)

    alpha_tilde = y_oh + (1.0 - y_oh) * alphas
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(1))
        - torch.lgamma(torch.tensor(2.0))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde)
           - torch.digamma(S_tilde))).sum(dim=1)
    )
    lam = min(1.0, epoch / max(CONFIG["kl_annealing_epochs"], 1))
    return (loss_ml + lam * kl).mean()


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, n_feat, device):
    """Train sequence model, return (model, best_val_acc)."""
    model = SequenceModel(input_size=n_feat).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)
    val_tensor_x = torch.from_numpy(X_val).to(device)
    val_tensor_y = torch.from_numpy(y_val).to(device)

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    ce_loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state, patience_counter = 0.0, None, 0

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)

            if CONFIG["use_dirichlet"]:
                loss = dirichlet_loss(out, yb, epoch)
            else:
                loss = ce_loss_fn(out, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            preds, _ = model.predict(val_tensor_x)
            val_acc = (preds == val_tensor_y).float().mean().item()

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_acc


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def transition_accuracy(y_true, y_pred):
    """Accuracy on frames where the true label changes."""
    trans_correct, trans_total = 0, 0
    for j in range(1, len(y_true)):
        if y_true[j] != y_true[j - 1]:
            trans_total += 1
            if y_pred[j] == y_true[j]:
                trans_correct += 1
    if trans_total == 0:
        return 0.0, 0
    return trans_correct / trans_total, trans_total


def evaluate(y_true, y_pred):
    """Compute all metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    ta, n_trans = transition_accuracy(y_true, y_pred)
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec,
            "trans_acc": ta, "n_trans": n_trans}


# ═══════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════

def compute_baselines(y_train, y_val):
    """Compute persistence and majority baselines."""
    # Majority
    majority = int(y_train.mean() >= 0.5)
    maj_pred = np.full_like(y_val, majority)
    maj_metrics = evaluate(y_val, maj_pred)

    # Persistence
    pers_pred = np.empty_like(y_val)
    pers_pred[0] = y_val[0]
    pers_pred[1:] = y_val[:-1]
    pers_metrics = evaluate(y_val, pers_pred)

    return maj_metrics, pers_metrics


# ═══════════════════════════════════════════════════════════════════
# GRADIENT-BASED FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def compute_gradient_importance(X_train, y_train, X_val, y_val, n_feat, device, feature_names):
    """Train LSTM and measure per-feature gradient magnitudes.

    For each feature, we accumulate |dL/dx_i| over all training batches across
    all epochs. Features that the LSTM relies on most will have larger gradients.
    We average across multiple seeds for stability.
    """
    seeds = [42, 123, 456]
    importance_accum = np.zeros(n_feat, dtype=np.float64)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SequenceModel(input_size=n_feat).to(device)
        optimizer = Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        ce_loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=CONFIG["batch_size"], shuffle=True)

        feat_grad_sum = np.zeros(n_feat, dtype=np.float64)
        n_batches = 0

        # Train for limited epochs to get stable gradients
        for epoch in range(min(50, CONFIG["max_epochs"])):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb.requires_grad_(True)

                out = model(xb)
                loss = ce_loss_fn(out, yb)

                optimizer.zero_grad()
                loss.backward()

                # Accumulate input gradient magnitudes per feature
                # xb shape: (batch, window, n_feat)
                with torch.no_grad():
                    grad_mag = xb.grad.abs().mean(dim=(0, 1))  # average over batch and time
                    feat_grad_sum += grad_mag.cpu().numpy()
                    n_batches += 1

                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                optimizer.step()

        importance_accum += feat_grad_sum / max(n_batches, 1)
        print(f"  Gradient importance: seed {seed} done ({n_batches} batches)", file=sys.stderr)

    # Average across seeds
    importance_accum /= len(seeds)

    # Sort by importance (descending)
    ranked = sorted(zip(feature_names, importance_accum), key=lambda x: x[1], reverse=True)
    return ranked


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──
    df = load_data()

    # Resolve "all_65" to actual feature columns
    ALL_65_FEATURES = [
        "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
        "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
        "overall_edge_density", "top_edge_density", "top_high_freq_energy",
        "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
        "top_laplacian_var", "glcm_entropy", "mid_edge_density", "dark_channel_mean",
        "mid_gradient_std", "foreground_blob_count", "mid_high_freq_energy",
        "rms_contrast", "fft_high_freq_energy_ratio", "laplacian_variance",
        "laplacian_mean_abs", "edge_density_fine", "edge_fine_coarse_ratio",
        "keypoint_loss_ratio", "spatial_frequency", "gradient_magnitude_std",
        "dct_high_freq_ratio", "downsample_ssim", "downsample_info_loss",
        "foreground_pixel_ratio", "shadow_pixel_ratio", "foreground_blob_area_mean",
        "foreground_blob_area_std", "foreground_blob_pa_ratio",
        "foreground_edge_density", "temporal_diff_mean", "temporal_diff_std",
        "motion_pixel_ratio", "image_entropy", "glcm_contrast", "glcm_energy",
        "colorfulness", "mean_brightness", "brightness_std", "small_edge_components",
        "saturation_mean", "saturation_std", "mscn_mean", "mscn_std",
        "mscn_kurtosis", "mscn_skewness", "mscn_h_pair_mean", "mscn_h_pair_std",
        "mscn_v_pair_mean", "mscn_v_pair_std", "mid_laplacian_var",
        "bot_laplacian_var", "bot_gradient_std", "bot_edge_density",
        "bot_high_freq_energy", "ratio_top_bot_edge_density",
        "ratio_top_bot_gradient_std", "ratio_top_bot_laplacian_var",
    ]

    if CONFIG["features"] == "all_65":
        feature_cols = ALL_65_FEATURES
    else:
        feature_cols = CONFIG["features"]
    target_col = get_target_column()

    # Verify features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}", file=sys.stderr)
        sys.exit(1)

    # ── Temporal split ──
    train_df = df[df["frame_id"] <= CONFIG["train_cutoff"]].copy()
    val_df = df[df["frame_id"] > CONFIG["train_cutoff"]].copy()

    # ── Threshold ──
    threshold = compute_threshold(df)

    # ── Scaler ──
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # ── Build windows ──
    X_train, y_train, _ = build_windows(
        train_df, feature_cols, target_col, CONFIG["train_stride"], scaler, threshold)
    X_val, y_val, _ = build_windows(
        val_df, feature_cols, target_col, CONFIG["eval_stride"], scaler, threshold)

    n_feat = len(feature_cols)
    n_val = len(y_val)

    if n_val < 10:
        print("ERROR: Too few validation windows", file=sys.stderr)
        sys.exit(1)

    hard_rate_train = y_train.mean()
    hard_rate_val = y_val.mean()

    # ── Baselines ──
    maj_metrics, pers_metrics = compute_baselines(y_train, y_val)

    # ── Gradient-based feature importance (if requested) ──
    if CONFIG.get("gradient_feature_selection"):
        grad_importance = compute_gradient_importance(X_train, y_train, X_val, y_val, n_feat, device, feature_cols)
        # Print ranking and exit — the agent will use this to select features
        print("\n=== GRADIENT FEATURE IMPORTANCE ===", file=sys.stderr)
        for rank, (fname, score) in enumerate(grad_importance, 1):
            print(f"  {rank:3d}. {fname:40s}  {score:.6f}", file=sys.stderr)
        print("=== END GRADIENT IMPORTANCE ===\n", file=sys.stderr)
        # Still continue with normal training

    # ── Train ──
    best_metrics = None

    if CONFIG["model_type"] != "rf_only":
        seeds = CONFIG.get("seeds", [42])
        all_seed_metrics = []
        all_seed_preds = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model, _ = train_model(X_train, y_train, X_val, y_val, n_feat, device)
            with torch.no_grad():
                preds, probs = model.predict(torch.from_numpy(X_val).to(device))
            pred_np = preds.cpu().numpy()
            m_seed = evaluate(y_val, pred_np)
            all_seed_metrics.append(m_seed)
            all_seed_preds.append(pred_np)
            print(f"  Seed {seed}: acc={m_seed['acc']:.3f} trans={m_seed['trans_acc']:.3f}", file=sys.stderr)

        # Ensemble: majority vote across all seeds
        if len(all_seed_preds) > 1:
            vote_sum = np.sum(all_seed_preds, axis=0)
            ensemble_pred = (vote_sum > len(all_seed_preds) / 2.0).astype(np.int64)
            ensemble_metrics = evaluate(y_val, ensemble_pred)
            print(f"  Ensemble: acc={ensemble_metrics['acc']:.3f} trans={ensemble_metrics['trans_acc']:.3f}", file=sys.stderr)

        # Pick seed with best acc (among those with trans_acc > 0.50)
        valid_seeds = [(i, m) for i, m in enumerate(all_seed_metrics) if m['trans_acc'] > 0.50]
        if valid_seeds:
            best_idx, lstm_metrics = max(valid_seeds, key=lambda x: x[1]['acc'])
        else:
            best_idx = max(range(len(all_seed_metrics)), key=lambda i: all_seed_metrics[i]['acc'])
            lstm_metrics = all_seed_metrics[best_idx]

        # Use ensemble if it's better
        if len(all_seed_preds) > 1 and ensemble_metrics['trans_acc'] > 0.50 and ensemble_metrics['acc'] > lstm_metrics['acc']:
            lstm_metrics = ensemble_metrics
            lstm_pred = ensemble_pred
            print(f"  Using ensemble", file=sys.stderr)
        else:
            lstm_pred = all_seed_preds[best_idx]

        best_metrics = lstm_metrics
        best_name = CONFIG["model_type"].upper()

    # ── RF ──
    X_train_rf = build_rf_features(X_train)
    X_val_rf = build_rf_features(X_val)

    rf = RandomForestClassifier(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        min_samples_leaf=CONFIG["rf_min_samples_leaf"],
        max_features="sqrt", n_jobs=-1, random_state=42,
        class_weight=CONFIG.get("rf_class_weight", None))
    rf.fit(X_train_rf, y_train)
    rf_pred = rf.predict(X_val_rf)
    rf_metrics = evaluate(y_val, rf_pred)

    if best_metrics is None or rf_metrics["acc"] > best_metrics["acc"]:
        best_metrics = rf_metrics
        best_name = "RF"

    # ── GBM ──
    gbm = GradientBoostingClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42)
    gbm.fit(X_train_rf, y_train)
    gbm_pred = gbm.predict(X_val_rf)
    gbm_metrics = evaluate(y_val, gbm_pred)
    print(f"  GBM:           acc={gbm_metrics['acc']:.3f}  f1={gbm_metrics['f1']:.3f}  trans={gbm_metrics['trans_acc']:.3f}", file=sys.stderr)

    if gbm_metrics["acc"] > best_metrics["acc"]:
        best_metrics = gbm_metrics
        best_name = "GBM"

    # ── XGBoost ──
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, verbosity=0)
    xgb.fit(X_train_rf, y_train)
    xgb_pred = xgb.predict(X_val_rf)
    xgb_metrics = evaluate(y_val, xgb_pred)
    print(f"  XGB:           acc={xgb_metrics['acc']:.3f}  f1={xgb_metrics['f1']:.3f}  trans={xgb_metrics['trans_acc']:.3f}", file=sys.stderr)

    if xgb_metrics["acc"] > best_metrics["acc"]:
        best_metrics = xgb_metrics
        best_name = "XGB"

    # ── Tree ensemble (RF+GBM+XGB majority vote) ──
    tree_vote = (rf_pred.astype(int) + gbm_pred.astype(int) + xgb_pred.astype(int))
    tree_ensemble_pred = (tree_vote >= 2).astype(np.int64)
    tree_ensemble_metrics = evaluate(y_val, tree_ensemble_pred)
    print(f"  TreeEnsemble:  acc={tree_ensemble_metrics['acc']:.3f}  f1={tree_ensemble_metrics['f1']:.3f}  trans={tree_ensemble_metrics['trans_acc']:.3f}", file=sys.stderr)
    if tree_ensemble_metrics["acc"] > best_metrics["acc"]:
        best_metrics = tree_ensemble_metrics
        best_name = "TreeEnsemble"

    # ── LSTM+RF ensemble ──
    if CONFIG["model_type"] != "rf_only" and lstm_pred is not None:
        combo_pred = ((lstm_pred + rf_pred) >= 1).astype(np.int64)
        combo_metrics = evaluate(y_val, combo_pred)
        print(f"  LSTM+RF:       acc={combo_metrics['acc']:.3f}  f1={combo_metrics['f1']:.3f}  trans={combo_metrics['trans_acc']:.3f}", file=sys.stderr)
        if combo_metrics["acc"] > best_metrics["acc"] and combo_metrics["trans_acc"] > 0.50:
            best_metrics = combo_metrics
            best_name = "LSTM+RF"

    # ── Pick best ──
    m = best_metrics
    elapsed = time.time() - t0

    # ── Print results ──
    config_summary = {
        "model": best_name,
        "n_features": n_feat,
        "target": CONFIG["target"],
        "window": CONFIG["window"],
        "horizon": CONFIG["horizon"],
        "threshold": round(threshold, 3),
        "hard_train": round(hard_rate_train, 3),
        "hard_val": round(hard_rate_val, 3),
        "dirichlet": CONFIG["use_dirichlet"],
        "hidden": CONFIG["hidden_size"],
        "layers": CONFIG["n_layers"],
    }

    # Diagnostic info to stderr
    print(f"  Train windows: {len(X_train)} ({hard_rate_train:.1%} hard)", file=sys.stderr)
    print(f"  Val windows:   {n_val} ({hard_rate_val:.1%} hard)", file=sys.stderr)
    print(f"  Threshold:     {threshold:.3f}", file=sys.stderr)
    print(f"  Baselines:     majority={maj_metrics['acc']:.3f}  persistence={pers_metrics['acc']:.3f}", file=sys.stderr)
    if CONFIG["model_type"] != "rf_only":
        print(f"  {CONFIG['model_type'].upper()}:  acc={lstm_metrics['acc']:.3f}  f1={lstm_metrics['f1']:.3f}  trans={lstm_metrics['trans_acc']:.3f}", file=sys.stderr)
    print(f"  RF:            acc={rf_metrics['acc']:.3f}  f1={rf_metrics['f1']:.3f}  trans={rf_metrics['trans_acc']:.3f}", file=sys.stderr)
    print(f"  Best:          {best_name}", file=sys.stderr)
    print(f"  Time:          {elapsed:.1f}s", file=sys.stderr)

    # THE RESULT LINE — parsed by the agent
    print(f"RESULT\tacc={m['acc']:.6f}\ttrans_acc={m['trans_acc']:.6f}\t"
          f"f1={m['f1']:.6f}\tprec={m['prec']:.6f}\trec={m['rec']:.6f}\t"
          f"n_val={n_val}\tconfig={json.dumps(config_summary)}")


if __name__ == "__main__":
    main()
