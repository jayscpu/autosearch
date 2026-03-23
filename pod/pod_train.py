#!/usr/bin/env python3
"""ECHO Bellevue — Multi-intersection regression + evidential regression.

Loads pre-extracted features, trains on intersections 1+2, evaluates on:
  1. Within-camera val (temporal split of train intersections)
  2. Cross-camera val (unseen intersection 3)
  3. Combined val

Trains 4 models every run, picks the best by mse_within:
  1. LSTM — MSE loss, multi-seed ensemble
  2. RF — RandomForestRegressor on summary features
  3. LSTM+RF — averaged predictions from 1 and 2
  4. EvidentialLSTM — NIG loss with uncertainty quantification

Prints RESULT line parsed by the agent.
"""

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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pod_features import ALL_FEATURES, TOP_35_SPEARMAN

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
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

    # ── Intersections ──
    "train_intersections": [
        "Bellevue_150th_Eastgate",
        "Bellevue_150th_Newport",
        "Bellevue_Bellevue_NE8th",
        "Bellevue_116th_NE12th",
    ],
    "test_intersection": "Bellevue_150th_SE38th",
    "train_fraction": 0.6,

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
}

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "pod_features_all.csv"


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(FEATURES_CSV)
    if CONFIG["target"] == "miss_rate":
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
    """Split into train, within-camera val, and cross-camera val."""
    train_ints = CONFIG["train_intersections"]
    test_int = CONFIG["test_intersection"]
    frac = CONFIG["train_fraction"]

    train_dfs = []
    within_val_dfs = []
    for int_name in train_ints:
        sub = df[df["sequence"] == int_name].sort_values("frame_id")
        cutoff_idx = int(len(sub) * frac)
        cutoff_fid = sub.iloc[cutoff_idx]["frame_id"]
        train_dfs.append(sub[sub["frame_id"] <= cutoff_fid])
        within_val_dfs.append(sub[sub["frame_id"] > cutoff_fid])

    train_df = pd.concat(train_dfs, ignore_index=True)
    within_val_df = pd.concat(within_val_dfs, ignore_index=True)
    cross_val_df = df[df["sequence"] == test_int].copy()

    return train_df, within_val_df, cross_val_df


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

    for _, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

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

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)  # shape: (N, n_steps)
    return X, y


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

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, n_steps),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # (batch, n_steps)


# ═══════════════════════════════════════════════════════════════════
# EVIDENTIAL LSTM (NIG output for uncertainty quantification)
# ═══════════════════════════════════════════════════════════════════

class EvidentialLSTM(nn.Module):
    """LSTM with Normal-Inverse-Gamma output for evidential regression.
    Outputs NIG params (γ, ν, α, β) for each of n_steps sub-windows."""

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
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, n_steps * 4),  # 4 NIG params per step
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        raw = self.head(h_n[-1])  # (batch, n_steps * 4)
        raw = raw.view(-1, self.n_steps, 4)

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
    evidence = 2.0 * nu + alpha
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
        scheduler.step()

        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

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

def transition_accuracy_3class(y_true, y_pred):
    """Accuracy on frames where the true 3-class label changes."""
    correct, total = 0, 0
    for j in range(1, len(y_true)):
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


def eval_regression(y_true_mean, y_pred_mean, t1, t2):
    """Evaluate regression predictions (plain mode or evidential gamma)."""
    mse = float(mean_squared_error(y_true_mean, y_pred_mean))
    mae = float(mean_absolute_error(y_true_mean, y_pred_mean))

    true_cls = to_3class(y_true_mean, t1, t2)
    pred_cls = to_3class(y_pred_mean, t1, t2)
    cls_acc = float(np.mean(pred_cls == true_cls))
    cls_trans, n_trans = transition_accuracy_3class(true_cls, pred_cls)

    return {"mse": mse, "mae": mae, "cls_acc": cls_acc,
            "cls_trans_acc": cls_trans, "n_trans": n_trans}


def eval_evidential(model, X, y, device, t1, t2):
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

    probs = np.stack([p_easy, p_moderate, p_hard], axis=1)
    pred_cls = probs.argmax(axis=1)
    true_cls = to_3class(y_mean, t1, t2)

    cls_acc = float(np.mean(pred_cls == true_cls))
    cls_trans, n_trans = transition_accuracy_3class(true_cls, pred_cls)

    # Uncertainty calibration
    correct_mask = (pred_cls == true_cls)
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        unc_separation = float(epistemic_mean[~correct_mask].mean()
                               - epistemic_mean[correct_mask].mean())
    else:
        unc_separation = 0.0

    return {"mse": mse, "mae": mae, "cls_acc": cls_acc,
            "cls_trans_acc": cls_trans, "n_trans": n_trans,
            "unc_separation": unc_separation}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU!", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Split
    train_df, within_val_df, cross_val_df = split_data(df)
    print(f"  Multi-step: horizon={H}, sub_window={S}, n_steps={n_steps}", file=sys.stderr)
    print(f"  Train: {len(train_df)} frames from {CONFIG['train_intersections']}",
          file=sys.stderr)
    print(f"  Within-val: {len(within_val_df)} frames", file=sys.stderr)
    print(f"  Cross-val: {len(cross_val_df)} frames from {CONFIG['test_intersection']}",
          file=sys.stderr)

    # Scaler from training data only
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # Build windows
    X_train, y_train = build_windows(
        train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_within, y_within = build_windows(
        within_val_df, feature_cols, CONFIG["eval_stride"], scaler)
    X_cross, y_cross = build_windows(
        cross_val_df, feature_cols, CONFIG["eval_stride"], scaler)

    n_feat = len(feature_cols)
    y_within_mean = y_within.mean(axis=1)
    y_cross_mean = y_cross.mean(axis=1)
    y_combined_mean = np.concatenate([y_within_mean, y_cross_mean])

    print(f"  Windows: train={len(y_train)}, within={len(y_within)}, "
          f"cross={len(y_cross)}", file=sys.stderr)

    if len(y_within) == 0 or len(y_cross) == 0:
        print("ERROR: Empty val set. Check data or window/horizon settings.",
              file=sys.stderr)
        sys.exit(1)

    # Difficulty thresholds from training data
    t1 = float(np.percentile(y_train.flatten(), CONFIG["t1_percentile"]))
    t2 = float(np.percentile(y_train.flatten(), CONFIG["t2_percentile"]))
    print(f"  Thresholds: t1={t1:.4f} (p{CONFIG['t1_percentile']}), "
          f"t2={t2:.4f} (p{CONFIG['t2_percentile']})", file=sys.stderr)

    # ═════════════════════════════════════════════════════════════
    # MODEL 1: LSTM (MSE loss, multi-seed ensemble)
    # ═════════════════════════════════════════════════════════════
    print("\n  ── LSTM ──", file=sys.stderr)
    all_preds_within, all_preds_cross = [], []
    for seed in CONFIG["seeds"]:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = train_plain(X_train, y_train, X_within, y_within,
                            n_feat, n_steps, device)
        with torch.no_grad():
            pw = model(torch.from_numpy(X_within).to(device)).cpu().numpy()
            pc = model(torch.from_numpy(X_cross).to(device)).cpu().numpy()
        all_preds_within.append(pw)
        all_preds_cross.append(pc)

    lstm_within_pred = np.mean(all_preds_within, axis=0).mean(axis=1)
    lstm_cross_pred = np.mean(all_preds_cross, axis=0).mean(axis=1)

    lstm_m_within = eval_regression(y_within_mean, lstm_within_pred, t1, t2)
    lstm_m_cross = eval_regression(y_cross_mean, lstm_cross_pred, t1, t2)
    lstm_comb_pred = np.concatenate([lstm_within_pred, lstm_cross_pred])
    lstm_m_combined = eval_regression(y_combined_mean, lstm_comb_pred, t1, t2)

    print(f"  LSTM within: mse={lstm_m_within['mse']:.4f} "
          f"cls_acc={lstm_m_within['cls_acc']:.3f}", file=sys.stderr)
    print(f"  LSTM cross:  mse={lstm_m_cross['mse']:.4f} "
          f"cls_acc={lstm_m_cross['cls_acc']:.3f}", file=sys.stderr)

    # ═════════════════════════════════════════════════════════════
    # MODEL 2: RF (regressor on summary features)
    # ═════════════════════════════════════════════════════════════
    print("\n  ── RF ──", file=sys.stderr)
    X_train_rf = build_rf_features(X_train)
    X_within_rf = build_rf_features(X_within)
    X_cross_rf = build_rf_features(X_cross)
    y_train_mean = y_train.mean(axis=1)

    rf = RandomForestRegressor(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        min_samples_leaf=CONFIG["rf_min_samples_leaf"],
        max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train_mean)

    rf_within_pred = rf.predict(X_within_rf)
    rf_cross_pred = rf.predict(X_cross_rf)

    rf_m_within = eval_regression(y_within_mean, rf_within_pred, t1, t2)
    rf_m_cross = eval_regression(y_cross_mean, rf_cross_pred, t1, t2)
    rf_comb_pred = np.concatenate([rf_within_pred, rf_cross_pred])
    rf_m_combined = eval_regression(y_combined_mean, rf_comb_pred, t1, t2)

    print(f"  RF within:   mse={rf_m_within['mse']:.4f} "
          f"cls_acc={rf_m_within['cls_acc']:.3f}", file=sys.stderr)
    print(f"  RF cross:    mse={rf_m_cross['mse']:.4f} "
          f"cls_acc={rf_m_cross['cls_acc']:.3f}", file=sys.stderr)

    # ═════════════════════════════════════════════════════════════
    # MODEL 3: LSTM+RF Ensemble (average predictions)
    # ═════════════════════════════════════════════════════════════
    print("\n  ── LSTM+RF ──", file=sys.stderr)
    ens_within_pred = 0.5 * lstm_within_pred + 0.5 * rf_within_pred
    ens_cross_pred = 0.5 * lstm_cross_pred + 0.5 * rf_cross_pred

    ens_m_within = eval_regression(y_within_mean, ens_within_pred, t1, t2)
    ens_m_cross = eval_regression(y_cross_mean, ens_cross_pred, t1, t2)
    ens_comb_pred = np.concatenate([ens_within_pred, ens_cross_pred])
    ens_m_combined = eval_regression(y_combined_mean, ens_comb_pred, t1, t2)

    print(f"  ENS within:  mse={ens_m_within['mse']:.4f} "
          f"cls_acc={ens_m_within['cls_acc']:.3f}", file=sys.stderr)
    print(f"  ENS cross:   mse={ens_m_cross['mse']:.4f} "
          f"cls_acc={ens_m_cross['cls_acc']:.3f}", file=sys.stderr)

    # ═════════════════════════════════════════════════════════════
    # MODEL 4: EvidentialLSTM (NIG loss, uncertainty)
    # ═════════════════════════════════════════════════════════════
    # Single seed for evidential: NIG training is ~3x slower than MSE,
    # multi-seed would exceed the 5-min-per-experiment budget.
    print("\n  ── EvidentialLSTM ──", file=sys.stderr)
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    evid_model = train_evidential(X_train, y_train, X_within, y_within,
                                  n_feat, n_steps, device)

    evid_m_within = eval_evidential(evid_model, X_within, y_within, device, t1, t2)
    evid_m_cross = eval_evidential(evid_model, X_cross, y_cross, device, t1, t2)
    X_comb = np.concatenate([X_within, X_cross])
    y_comb = np.concatenate([y_within, y_cross])
    evid_m_combined = eval_evidential(evid_model, X_comb, y_comb, device, t1, t2)

    print(f"  EVID within: mse={evid_m_within['mse']:.4f} "
          f"cls_acc={evid_m_within['cls_acc']:.3f} "
          f"unc_sep={evid_m_within['unc_separation']:.4f}", file=sys.stderr)
    print(f"  EVID cross:  mse={evid_m_cross['mse']:.4f} "
          f"cls_acc={evid_m_cross['cls_acc']:.3f} "
          f"unc_sep={evid_m_cross['unc_separation']:.4f}", file=sys.stderr)

    # ═════════════════════════════════════════════════════════════
    # PICK BEST by mse_within, subject to cls_trans > 0.50
    # ═════════════════════════════════════════════════════════════
    candidates = [
        ("LSTM", lstm_m_within, lstm_m_cross, lstm_m_combined),
        ("RF", rf_m_within, rf_m_cross, rf_m_combined),
        ("LSTM+RF", ens_m_within, ens_m_cross, ens_m_combined),
        ("EvidentialLSTM", evid_m_within, evid_m_cross, evid_m_combined),
    ]

    # Filter: cls_trans > 0.50 on both within and cross val sets
    valid = [c for c in candidates
             if c[1]["cls_trans_acc"] > 0.50 and c[2]["cls_trans_acc"] > 0.50]
    if not valid:
        # Fallback: pick from all if none pass the filter
        print("  WARNING: No model passed cls_trans > 0.50, picking best MSE anyway",
              file=sys.stderr)
        valid = candidates

    best_name, best_within, best_cross, best_combined = min(
        valid, key=lambda x: x[1]["mse"])

    elapsed = time.time() - t0

    # Summary table
    print(f"\n  {'Model':<18} {'MSE_w':>8} {'MSE_x':>8} {'cls_acc_w':>10} {'cls_acc_x':>10}",
          file=sys.stderr)
    print(f"  {'-'*56}", file=sys.stderr)
    for name, mw, mx, _ in candidates:
        marker = " *" if name == best_name else ""
        print(f"  {name:<18} {mw['mse']:>8.4f} {mx['mse']:>8.4f} "
              f"{mw['cls_acc']:>10.3f} {mx['cls_acc']:>10.3f}{marker}",
              file=sys.stderr)
    print(f"\n  Best: {best_name}  Time: {elapsed:.1f}s", file=sys.stderr)

    config_summary = {
        "model": best_name,
        "n_features": n_feat,
        "window": CONFIG["window"],
        "horizon": CONFIG["horizon"],
        "sub_window": CONFIG["sub_window"],
        "n_steps": n_steps,
        "lambda1": CONFIG["lambda1"],
        "train_ints": CONFIG["train_intersections"],
        "test_int": CONFIG["test_intersection"],
        "t1": round(t1, 4),
        "t2": round(t2, 4),
    }

    # RESULT LINE (best model) — fixed fields for consistent parsing
    unc_within = best_within.get("unc_separation", 0.0)
    unc_cross = best_cross.get("unc_separation", 0.0)

    print("\t".join([
        "RESULT",
        f"model={best_name}",
        f"mse_within={best_within['mse']:.6f}",
        f"mse_cross={best_cross['mse']:.6f}",
        f"mse_combined={best_combined['mse']:.6f}",
        f"mae_within={best_within['mae']:.6f}",
        f"mae_cross={best_cross['mae']:.6f}",
        f"cls_acc_within={best_within['cls_acc']:.6f}",
        f"cls_acc_cross={best_cross['cls_acc']:.6f}",
        f"cls_trans_within={best_within['cls_trans_acc']:.6f}",
        f"cls_trans_cross={best_cross['cls_trans_acc']:.6f}",
        f"unc_sep_within={unc_within:.6f}",
        f"unc_sep_cross={unc_cross:.6f}",
        f"n_within={len(y_within)}",
        f"n_cross={len(y_cross)}",
        f"config={json.dumps(config_summary)}",
    ]))


if __name__ == "__main__":
    main()
