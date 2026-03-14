#!/usr/bin/env python3
"""ECHO Evidential Regression — NIG-based miss_rate prediction with uncertainty.

Predicts continuous miss_rate using Normal-Inverse-Gamma evidential learning.
Multi-step: outputs NIG params (γ, ν, α, β) for EACH sub-window in the horizon.
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
    "lambda1": 0.3,          # evidence regularizer weight
    "lambda2": 0.05,         # uncertainty regularizer weight

    # ── Difficulty Thresholds (percentiles of training miss_rate) ──
    "t1_percentile": 10,     # easy/moderate boundary
    "t2_percentile": 85,     # moderate/hard boundary

    # ── Windowing ──
    "window": 30,
    "horizon": 30,
    "sub_window": 6,         # multi-step: each step predicts mean over sub_window frames
    "train_stride": 10,
    "eval_stride": 30,

    # ── Temporal split ──
    "train_cutoff": 14400,

    # ── Architecture ──
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
# WINDOWING (multi-step: N_steps targets per sample)
# ═══════════════════════════════════════════════════════════════════

def build_windows(df, feature_cols, stride, scaler):
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S
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

            # Multi-step targets: mean miss_rate for each sub-window
            targets = np.zeros(n_steps, dtype=np.float32)
            for step in range(n_steps):
                start = t + W + step * S
                end = start + S
                targets[step] = target_vals[start:end].mean()

            X_list.append(x_window)
            y_list.append(targets)
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)  # shape: (N, n_steps)
    return X, y, seq_ids


# ═══════════════════════════════════════════════════════════════════
# EVIDENTIAL LSTM MODEL (multi-step output)
# ═══════════════════════════════════════════════════════════════════

class EvidentialLSTM(nn.Module):
    """LSTM with Normal-Inverse-Gamma output for evidential regression.
    Outputs NIG params for each of n_steps sub-windows."""

    def __init__(self, input_size, n_steps):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0
        self.n_steps = n_steps

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, n_steps * 4),  # 4 NIG params per step
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        raw = self.head(h_n[-1])  # (batch, n_steps * 4)
        raw = raw.view(-1, self.n_steps, 4)  # (batch, n_steps, 4)

        gamma = raw[:, :, 0]                              # predicted mean
        nu = self.softplus(raw[:, :, 1]) + 1e-6            # ν > 0
        alpha = self.softplus(raw[:, :, 2]) + 1.0 + 1e-6   # α > 1
        beta = self.softplus(raw[:, :, 3]) + 1e-6           # β > 0

        return gamma, nu, alpha, beta  # each (batch, n_steps)

    def predict(self, x):
        gamma, nu, alpha, beta = self.forward(x)
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return gamma, aleatoric, epistemic, nu, alpha, beta


# ═══════════════════════════════════════════════════════════════════
# NIG LOSS FUNCTION (Amini et al. 2020) — multi-step
# ═══════════════════════════════════════════════════════════════════

def nig_loss(gamma, nu, alpha, beta, y):
    """Compute NIG evidential regression loss, averaged over steps."""
    # gamma, nu, alpha, beta, y: all (batch, n_steps)
    omega = 2.0 * beta * (1.0 + nu)
    nll = (0.5 * torch.log(np.pi / nu)
           - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))

    error = torch.abs(y - gamma)
    evidence = 2.0 * nu + alpha
    reg_evidence = error * evidence

    reg_uncertainty = (1.0 / (alpha - 1.0 + 1e-6))

    loss = nll + CONFIG["lambda1"] * reg_evidence + CONFIG["lambda2"] * reg_uncertainty
    return loss.mean()  # mean over batch AND steps


# ═══════════════════════════════════════════════════════════════════
# STUDENT-T CDF CONVERSION TO 3-CLASS PROBABILITIES
# ═══════════════════════════════════════════════════════════════════

def nig_to_class_probs(gamma, nu, alpha, beta, t1, t2):
    """Convert NIG parameters to P(easy), P(moderate), P(hard) via Student-t CDF.
    Inputs can be 1D (single-step) or already numpy arrays."""
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
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, n_feat, n_steps, device):
    model = EvidentialLSTM(input_size=n_feat, n_steps=n_steps).to(device)
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
# EVALUATION (multi-step)
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
    """Full evaluation: regression metrics, uncertainty calibration, 3-class accuracy.
    Multi-step: evaluates averaged across all steps, and also per-step."""
    val_x = torch.from_numpy(X_val).to(device)
    n_steps = y_val.shape[1]

    with torch.no_grad():
        gamma, aleatoric, epistemic, nu, alpha, beta = model.predict(val_x)

    # Use mean across steps for the overall single-step-equivalent metrics
    gamma_mean = gamma.cpu().numpy().mean(axis=1)  # (N,)
    y_val_mean = y_val.mean(axis=1)  # (N,)
    epistemic_mean = epistemic.cpu().numpy().mean(axis=1)  # (N,)

    # ── Regression metrics (on step-averaged predictions) ──
    mse = float(np.mean((gamma_mean - y_val_mean) ** 2))
    mae = float(np.mean(np.abs(gamma_mean - y_val_mean)))

    # ── Per-step MSE ──
    gamma_np = gamma.cpu().numpy()
    per_step_mse = [float(np.mean((gamma_np[:, s] - y_val[:, s]) ** 2)) for s in range(n_steps)]

    # ── Difficulty thresholds from training data (use all training targets flattened) ──
    t1 = float(np.percentile(y_train_for_thresholds.flatten(), CONFIG["t1_percentile"]))
    t2 = float(np.percentile(y_train_for_thresholds.flatten(), CONFIG["t2_percentile"]))

    # ── 3-class conversion: use mean NIG params across steps ──
    gamma_avg = gamma.mean(dim=1).cpu()
    nu_avg = nu.mean(dim=1).cpu()
    alpha_avg = alpha.mean(dim=1).cpu()
    beta_avg = beta.mean(dim=1).cpu()

    p_easy, p_moderate, p_hard = nig_to_class_probs(gamma_avg, nu_avg, alpha_avg, beta_avg, t1, t2)

    probs = np.stack([p_easy, p_moderate, p_hard], axis=1)
    pred_class = probs.argmax(axis=1)

    true_class = np.zeros(len(y_val_mean), dtype=np.int64)
    true_class[y_val_mean >= t2] = 2
    true_class[(y_val_mean >= t1) & (y_val_mean < t2)] = 1

    cls_acc = float(np.mean(pred_class == true_class))
    cls_trans_acc, n_trans = transition_accuracy_3class(true_class, pred_class)

    # ── Uncertainty calibration ──
    correct_mask = (pred_class == true_class)
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        mean_ep_correct = float(epistemic_mean[correct_mask].mean())
        mean_ep_incorrect = float(epistemic_mean[~correct_mask].mean())
        unc_separation = mean_ep_incorrect - mean_ep_correct
    else:
        unc_separation = 0.0
        mean_ep_correct = float(epistemic_mean.mean())
        mean_ep_incorrect = float(epistemic_mean.mean())

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
        "per_step_mse": per_step_mse,
        "n_steps": n_steps,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()
    feature_cols = CONFIG["features"]

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}", file=sys.stderr)
        sys.exit(1)

    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S
    print(f"  Multi-step: horizon={H}, sub_window={S}, n_steps={n_steps}", file=sys.stderr)

    train_df = df[df["frame_id"] <= CONFIG["train_cutoff"]].copy()
    val_df = df[df["frame_id"] > CONFIG["train_cutoff"]].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    X_train, y_train, _ = build_windows(train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_val, y_val, _ = build_windows(val_df, feature_cols, CONFIG["eval_stride"], scaler)

    n_feat = len(feature_cols)
    n_val = len(y_val)

    if n_val < 10:
        print("ERROR: Too few validation windows", file=sys.stderr)
        sys.exit(1)

    print(f"  Train: {len(X_train)} windows, y shape={y_train.shape}", file=sys.stderr)
    print(f"  Val:   {n_val} windows, y shape={y_val.shape}", file=sys.stderr)

    model = train_model(X_train, y_train, X_val, y_val, n_feat, n_steps, device)
    metrics = evaluate(model, X_val, y_val, device, y_train)

    elapsed = time.time() - t0

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
    print(f"  Per-step MSE:   {[f'{m:.6f}' for m in metrics['per_step_mse']]}", file=sys.stderr)
    print(f"  Time:           {elapsed:.1f}s", file=sys.stderr)

    config_summary = {
        "n_features": n_feat,
        "lambda1": CONFIG["lambda1"],
        "lambda2": CONFIG["lambda2"],
        "t1_pct": CONFIG["t1_percentile"],
        "t2_pct": CONFIG["t2_percentile"],
        "window": CONFIG["window"],
        "horizon": CONFIG["horizon"],
        "sub_window": CONFIG["sub_window"],
        "n_steps": n_steps,
        "hidden": CONFIG["hidden_size"],
        "layers": CONFIG["n_layers"],
        "dropout": CONFIG["dropout"],
        "lr": CONFIG["lr"],
        "unc_sep": round(metrics["unc_separation"], 6),
    }

    print(f"RESULT\tmse={metrics['mse']:.6f}\tmae={metrics['mae']:.6f}\t"
          f"unc_separation={metrics['unc_separation']:.6f}\t"
          f"cls_acc={metrics['cls_acc']:.6f}\t"
          f"cls_trans_acc={metrics['cls_trans_acc']:.6f}\t"
          f"n_val={n_val}\tconfig={json.dumps(config_summary)}")


if __name__ == "__main__":
    main()
