#!/usr/bin/env python3
"""ECHO Evidential Regression — NIG-based miss_rate prediction with uncertainty.

Predicts continuous miss_rate using Normal-Inverse-Gamma evidential learning.
Outputs: prediction accuracy (MSE/MAE), uncertainty calibration, 3-class probabilities.
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
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — the agent modifies this section
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Features (top-35 Spearman from classification search) ──
    "features": [
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
    ],

    # ── NIG Loss Hyperparameters ──
    "lambda1": 1.0,          # evidence regularizer weight
    "lambda2": 0.05,         # uncertainty regularizer weight

    # ── Difficulty Thresholds (percentiles of training miss_rate) ──
    "t1_percentile": 33,     # easy/moderate boundary
    "t2_percentile": 67,     # moderate/hard boundary

    # ── Windowing ──
    "window": 30,
    "horizon": 30,
    "train_stride": 10,
    "eval_stride": 30,

    # ── Temporal split ──
    "train_cutoff": 14400,

    # ── Architecture ──
    "hidden_size": 64,
    "n_layers": 2,
    "dropout": 0.2,

    # ── Training ──
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 300,
    "patience": 40,
    "grad_clip": 1.0,
}

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "yolox_features.csv"
DETS_CSV = SCRIPT_DIR / "yolox_detections.csv"


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(FEATURES_CSV)
    dets = pd.read_csv(DETS_CSV)
    df = df.merge(dets[["frame_id", "nano_tp", "nano_fp", "x_count"]],
                  on="frame_id", how="left", suffixes=("", "_det"))
    for col in ["nano_tp", "nano_fp", "x_count"]:
        det_col = f"{col}_det"
        if det_col in df.columns:
            df[col] = df[det_col].fillna(df[col])
            df.drop(columns=[det_col], inplace=True)
    df["miss_rate"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    return df


# ═══════════════════════════════════════════════════════════════════
# WINDOWING (continuous target — mean miss_rate over horizon)
# ═══════════════════════════════════════════════════════════════════

def build_windows(df, feature_cols, stride, scaler):
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
        target_vals = sdf["miss_rate"].values

        for t in range(0, n - total_need + 1, stride):
            x_window = feat_vals[t: t + W]
            target_mean = target_vals[t + W: t + W + H].mean()

            X_list.append(x_window)
            y_list.append(target_mean)
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, seq_ids


# ═══════════════════════════════════════════════════════════════════
# EVIDENTIAL LSTM MODEL
# ═══════════════════════════════════════════════════════════════════

class EvidentialLSTM(nn.Module):
    """LSTM with Normal-Inverse-Gamma output for evidential regression."""

    def __init__(self, input_size):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(32, 4),  # outputs: gamma, log_nu, log_alpha_minus1, log_beta
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        raw = self.head(h_n[-1])

        gamma = raw[:, 0]                              # predicted mean (unconstrained)
        nu = self.softplus(raw[:, 1]) + 1e-6            # ν > 0
        alpha = self.softplus(raw[:, 2]) + 1.0 + 1e-6   # α > 1
        beta = self.softplus(raw[:, 3]) + 1e-6           # β > 0

        return gamma, nu, alpha, beta

    def predict(self, x):
        gamma, nu, alpha, beta = self.forward(x)
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return gamma, aleatoric, epistemic, nu, alpha, beta


# ═══════════════════════════════════════════════════════════════════
# NIG LOSS FUNCTION (Amini et al. 2020)
# ═══════════════════════════════════════════════════════════════════

def nig_loss(gamma, nu, alpha, beta, y):
    """Compute NIG evidential regression loss."""
    # Negative log-likelihood of Student-t marginal
    omega = 2.0 * beta * (1.0 + nu)
    nll = (0.5 * torch.log(np.pi / nu)
           - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))

    # Evidence regularizer: penalize confident wrong predictions
    error = torch.abs(y - gamma)
    evidence = 2.0 * nu + alpha
    reg_evidence = error * evidence

    # Uncertainty regularizer: stable gradients when α → 1
    reg_uncertainty = (1.0 / (alpha - 1.0 + 1e-6))

    loss = nll + CONFIG["lambda1"] * reg_evidence + CONFIG["lambda2"] * reg_uncertainty
    return loss.mean()


# ═══════════════════════════════════════════════════════════════════
# STUDENT-T CDF CONVERSION TO 3-CLASS PROBABILITIES
# ═══════════════════════════════════════════════════════════════════

def nig_to_class_probs(gamma, nu, alpha, beta, t1, t2):
    """Convert NIG parameters to P(easy), P(moderate), P(hard) via Student-t CDF."""
    gamma = gamma.cpu().numpy()
    nu = nu.cpu().numpy()
    alpha = alpha.cpu().numpy()
    beta = beta.cpu().numpy()

    # Student-t parameters
    df = 2.0 * alpha                        # degrees of freedom
    loc = gamma                              # location
    scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))  # scale

    # Compute CDF at thresholds
    p_easy = np.zeros(len(gamma))
    p_moderate = np.zeros(len(gamma))
    p_hard = np.zeros(len(gamma))

    for i in range(len(gamma)):
        if df[i] <= 0 or scale[i] <= 0 or np.isnan(df[i]) or np.isnan(scale[i]):
            # Fallback: uniform
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

        # Renormalize
        total = p_easy[i] + p_moderate[i] + p_hard[i]
        p_easy[i] /= total
        p_moderate[i] /= total
        p_hard[i] /= total

    return p_easy, p_moderate, p_hard


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, n_feat, device):
    model = EvidentialLSTM(input_size=n_feat).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)

    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"],
                     weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG["max_epochs"]):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0
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
            train_loss_sum += loss.item() * len(yb)
            train_count += len(yb)

        # Validate
        model.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = model(val_x)
            val_loss = nig_loss(gamma, nu, alpha, beta, val_y)

        val_loss_val = val_loss.item()
        scheduler.step(val_loss_val)

        if val_loss_val < best_val_loss and not np.isnan(val_loss_val):
            best_val_loss = val_loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            tl = train_loss_sum / max(train_count, 1)
            print(f"  Epoch {epoch:3d}: train_loss={tl:.4f} val_loss={val_loss_val:.4f}"
                  + (" *" if patience_counter == 0 else ""), file=sys.stderr)

        if patience_counter >= CONFIG["patience"]:
            print(f"  Early stop at epoch {epoch}", file=sys.stderr)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def transition_accuracy_3class(y_true, y_pred):
    """Accuracy on frames where the true 3-class label changes."""
    trans_correct, trans_total = 0, 0
    for j in range(1, len(y_true)):
        if y_true[j] != y_true[j - 1]:
            trans_total += 1
            if y_pred[j] == y_true[j]:
                trans_correct += 1
    if trans_total == 0:
        return 0.0, 0
    return trans_correct / trans_total, trans_total


def evaluate(model, X_val, y_val, device, y_train_for_thresholds):
    """Full evaluation: regression metrics, uncertainty calibration, 3-class accuracy."""
    val_x = torch.from_numpy(X_val).to(device)

    with torch.no_grad():
        gamma, aleatoric, epistemic, nu, alpha, beta = model.predict(val_x)

    gamma_np = gamma.cpu().numpy()
    aleatoric_np = aleatoric.cpu().numpy()
    epistemic_np = epistemic.cpu().numpy()

    # ── Regression metrics ──
    mse = float(np.mean((gamma_np - y_val) ** 2))
    mae = float(np.mean(np.abs(gamma_np - y_val)))

    # ── Difficulty thresholds from training data ──
    t1 = float(np.percentile(y_train_for_thresholds, CONFIG["t1_percentile"]))
    t2 = float(np.percentile(y_train_for_thresholds, CONFIG["t2_percentile"]))

    # ── 3-class conversion via Student-t CDF ──
    p_easy, p_moderate, p_hard = nig_to_class_probs(
        gamma.cpu(), nu.cpu(), alpha.cpu(), beta.cpu(), t1, t2)

    # Predicted class
    probs = np.stack([p_easy, p_moderate, p_hard], axis=1)
    pred_class = probs.argmax(axis=1)

    # True class
    true_class = np.zeros(len(y_val), dtype=np.int64)
    true_class[y_val >= t2] = 2  # hard
    true_class[(y_val >= t1) & (y_val < t2)] = 1  # moderate
    # easy = 0 (default)

    # ── 3-class accuracy ──
    cls_acc = float(np.mean(pred_class == true_class))

    # ── Transition accuracy ──
    cls_trans_acc, n_trans = transition_accuracy_3class(true_class, pred_class)

    # ── Uncertainty calibration ──
    correct_mask = (pred_class == true_class)
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        mean_ep_correct = float(epistemic_np[correct_mask].mean())
        mean_ep_incorrect = float(epistemic_np[~correct_mask].mean())
        unc_separation = mean_ep_incorrect - mean_ep_correct  # should be > 0
    else:
        unc_separation = 0.0
        mean_ep_correct = float(epistemic_np.mean())
        mean_ep_incorrect = float(epistemic_np.mean())

    # ── Per-class accuracy ──
    per_class = {}
    for c, name in enumerate(["easy", "moderate", "hard"]):
        mask = true_class == c
        if mask.sum() > 0:
            per_class[name] = float(np.mean(pred_class[mask] == c))
        else:
            per_class[name] = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "unc_separation": unc_separation,
        "mean_ep_correct": mean_ep_correct,
        "mean_ep_incorrect": mean_ep_incorrect,
        "cls_acc": cls_acc,
        "cls_trans_acc": cls_trans_acc,
        "n_trans": n_trans,
        "t1": t1,
        "t2": t2,
        "per_class": per_class,
        "class_dist": {
            "easy": int((true_class == 0).sum()),
            "moderate": int((true_class == 1).sum()),
            "hard": int((true_class == 2).sum()),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──
    df = load_data()
    feature_cols = CONFIG["features"]

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}", file=sys.stderr)
        sys.exit(1)

    # ── Temporal split ──
    train_df = df[df["frame_id"] <= CONFIG["train_cutoff"]].copy()
    val_df = df[df["frame_id"] > CONFIG["train_cutoff"]].copy()

    # ── Scaler ──
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # ── Build windows ──
    X_train, y_train, _ = build_windows(train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_val, y_val, _ = build_windows(val_df, feature_cols, CONFIG["eval_stride"], scaler)

    n_feat = len(feature_cols)
    n_val = len(y_val)

    if n_val < 10:
        print("ERROR: Too few validation windows", file=sys.stderr)
        sys.exit(1)

    print(f"  Train: {len(X_train)} windows, mean miss_rate={y_train.mean():.3f}", file=sys.stderr)
    print(f"  Val:   {n_val} windows, mean miss_rate={y_val.mean():.3f}", file=sys.stderr)

    # ── Train ──
    model = train_model(X_train, y_train, X_val, y_val, n_feat, device)

    # ── Evaluate ──
    metrics = evaluate(model, X_val, y_val, device, y_train)

    elapsed = time.time() - t0

    # ── Print diagnostics ──
    print(f"\n  === RESULTS ===", file=sys.stderr)
    print(f"  MSE:  {metrics['mse']:.6f}", file=sys.stderr)
    print(f"  MAE:  {metrics['mae']:.6f}", file=sys.stderr)
    print(f"  Unc separation: {metrics['unc_separation']:.6f} "
          f"(correct={metrics['mean_ep_correct']:.4f}, "
          f"incorrect={metrics['mean_ep_incorrect']:.4f})", file=sys.stderr)
    print(f"  3-class acc:    {metrics['cls_acc']:.3f}", file=sys.stderr)
    print(f"  3-class trans:  {metrics['cls_trans_acc']:.3f} "
          f"({metrics['n_trans']} transitions)", file=sys.stderr)
    print(f"  Per-class:      {metrics['per_class']}", file=sys.stderr)
    print(f"  Class dist:     {metrics['class_dist']}", file=sys.stderr)
    print(f"  Thresholds:     t1={metrics['t1']:.4f}, t2={metrics['t2']:.4f}", file=sys.stderr)
    print(f"  Time:           {elapsed:.1f}s", file=sys.stderr)

    # ── Config summary ──
    config_summary = {
        "n_features": n_feat,
        "lambda1": CONFIG["lambda1"],
        "lambda2": CONFIG["lambda2"],
        "t1_pct": CONFIG["t1_percentile"],
        "t2_pct": CONFIG["t2_percentile"],
        "window": CONFIG["window"],
        "horizon": CONFIG["horizon"],
        "hidden": CONFIG["hidden_size"],
        "layers": CONFIG["n_layers"],
        "dropout": CONFIG["dropout"],
        "lr": CONFIG["lr"],
        "unc_sep": round(metrics["unc_separation"], 6),
    }

    # ── THE RESULT LINE ──
    print(f"RESULT\tmse={metrics['mse']:.6f}\tmae={metrics['mae']:.6f}\t"
          f"unc_separation={metrics['unc_separation']:.6f}\t"
          f"cls_acc={metrics['cls_acc']:.6f}\t"
          f"cls_trans_acc={metrics['cls_trans_acc']:.6f}\t"
          f"n_val={n_val}\tconfig={json.dumps(config_summary)}")


if __name__ == "__main__":
    main()
