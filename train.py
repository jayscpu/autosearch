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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — the agent modifies this section
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Features ──
    "features": [
        "bot_edge_density", "bot_gradient_std", "bot_high_freq_energy",
        "bot_laplacian_var", "brightness_std", "colorfulness",
        "dark_channel_mean", "dct_high_freq_ratio", "downsample_info_loss",
        "downsample_ssim", "edge_density_coarse", "edge_density_fine",
        "edge_fine_coarse_ratio", "fast_keypoints_full", "fast_keypoints_half",
        "fft_critical_band_ratio", "fft_high_freq_energy_ratio",
        "foreground_blob_area_mean", "foreground_blob_area_std",
        "foreground_blob_count", "foreground_blob_pa_ratio",
        "foreground_edge_density", "foreground_pixel_ratio",
        "gabor_nyquist_energy", "glcm_contrast", "glcm_energy",
        "glcm_entropy", "glcm_homogeneity", "gradient_magnitude_mean",
        "gradient_magnitude_std", "image_entropy", "keypoint_loss_ratio",
        "laplacian_mean_abs", "laplacian_variance", "mean_brightness",
        "mid_edge_density", "mid_gradient_std", "mid_high_freq_energy",
        "mid_laplacian_var", "motion_pixel_ratio", "mscn_h_pair_mean",
        "mscn_h_pair_std", "mscn_kurtosis", "mscn_mean", "mscn_skewness",
        "mscn_std", "mscn_v_pair_mean", "mscn_v_pair_std",
        "overall_edge_density", "ratio_top_bot_edge_density",
        "ratio_top_bot_gradient_std", "ratio_top_bot_laplacian_var",
        "rms_contrast", "saturation_mean", "saturation_std",
        "shadow_pixel_ratio", "small_edge_components", "spatial_frequency",
        "temporal_diff_mean", "temporal_diff_std", "top_edge_density",
        "top_gradient_std", "top_high_freq_energy", "top_laplacian_var",
        "total_edge_components",
    ],

    # ── Target definition ──
    "target": "miss_rate",        # "fn_nano", "miss_rate", or "frame_f1"
    "threshold_method": "median",  # "median", "fixed", "percentile"
    "threshold_value": None,       # used if threshold_method="fixed"
    "min_threshold": None,          # floor for threshold (None for miss_rate)

    # ── Windowing ──
    "window": 30,
    "horizon": 30,
    "train_stride": 15,
    "eval_stride": 30,            # should equal window for non-overlapping val

    # ── Temporal split ──
    "train_cutoff": 14400,

    # ── Model ──
    "model_type": "lstm",          # "lstm", "gru", "rf_only"
    "hidden_size": 64,
    "n_layers": 2,
    "dropout": 0.2,
    "use_dirichlet": False,        # True=Dirichlet head, False=plain softmax+CE
    "kl_annealing_epochs": 10,

    # ── Training ──
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 25,
    "grad_clip": 1.0,

    # ── RF ──
    "rf_n_estimators": 500,
    "rf_max_depth": 20,
    "rf_min_samples_leaf": 10,
    "rf_features": "mean_std_slope",  # "mean_std_slope", "mean_std_slope_minmax", "all"
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
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──
    df = load_data()
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

    # ── Train ──
    best_metrics = None

    if CONFIG["model_type"] != "rf_only":
        model, _ = train_model(X_train, y_train, X_val, y_val, n_feat, device)

        with torch.no_grad():
            preds, probs = model.predict(torch.from_numpy(X_val).to(device))
        lstm_pred = preds.cpu().numpy()
        lstm_metrics = evaluate(y_val, lstm_pred)
        best_metrics = lstm_metrics
        best_name = CONFIG["model_type"].upper()

    # ── RF ──
    X_train_rf = build_rf_features(X_train)
    X_val_rf = build_rf_features(X_val)

    rf = RandomForestClassifier(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        min_samples_leaf=CONFIG["rf_min_samples_leaf"],
        max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train)
    rf_pred = rf.predict(X_val_rf)
    rf_metrics = evaluate(y_val, rf_pred)

    if best_metrics is None or rf_metrics["acc"] > best_metrics["acc"]:
        best_metrics = rf_metrics
        best_name = "RF"

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
