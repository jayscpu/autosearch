#!/usr/bin/env python3
"""Non-evidential regression comparison models.

Compares plain LSTM, GRU, RF, GradientBoosting, XGBoost, and LSTM+RF ensemble
against the evidential NIG model on the same continuous miss_rate target.
Uses MC dropout for uncertainty on neural models.
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
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
    "window": 30,
    "horizon": 30,
    "train_stride": 10,
    "eval_stride": 30,
    "train_cutoff": 14400,
    "t1_percentile": 25,
    "t2_percentile": 80,
    # Neural model params
    "hidden_size": 64,
    "n_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 300,
    "patience": 40,
    "grad_clip": 1.0,
    "mc_dropout_passes": 10,
}

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "yolox_features.csv"
DETS_CSV = SCRIPT_DIR / "yolox_detections.csv"


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


def build_windows(df, feature_cols, stride, scaler):
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    total_need = W + H
    X_list, y_list = [], []

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

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def build_flat_windows(df, feature_cols, stride, scaler):
    """Build windows with flattened features for tree models."""
    X, y = build_windows(df, feature_cols, stride, scaler)
    # Flatten: (N, W, F) -> (N, W*F)
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat, y


# ═══════════════════════════════════════════════════════════════════
# NEURAL MODELS
# ═══════════════════════════════════════════════════════════════════

class PlainLSTM(nn.Module):
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
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class PlainGRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden,
                           num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(-1)


def train_neural(model, X_train, y_train, X_val, y_val, device):
    model = model.to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def mc_dropout_predict(model, X, device, n_passes=10):
    """MC Dropout: run model with dropout ON multiple times, get mean + variance."""
    model.train()  # enable dropout
    x = torch.from_numpy(X).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x).cpu().numpy()
            preds.append(pred)
    preds = np.stack(preds, axis=0)  # (n_passes, N)
    mean_pred = preds.mean(axis=0)
    var_pred = preds.var(axis=0)
    model.eval()
    return mean_pred, var_pred


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(y_pred, y_val, uncertainty, y_train, model_name):
    """Evaluate a model on MSE, MAE, and 3-class accuracy."""
    mse = float(mean_squared_error(y_val, y_pred))
    mae = float(mean_absolute_error(y_val, y_pred))

    t1 = float(np.percentile(y_train, CONFIG["t1_percentile"]))
    t2 = float(np.percentile(y_train, CONFIG["t2_percentile"]))

    # 3-class from predictions
    pred_class = np.zeros(len(y_pred), dtype=int)
    pred_class[y_pred >= t2] = 2
    pred_class[(y_pred >= t1) & (y_pred < t2)] = 1

    true_class = np.zeros(len(y_val), dtype=int)
    true_class[y_val >= t2] = 2
    true_class[(y_val >= t1) & (y_val < t2)] = 1

    cls_acc = float(np.mean(pred_class == true_class))

    # Uncertainty separation
    correct_mask = (pred_class == true_class)
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        unc_sep = float(uncertainty[~correct_mask].mean() - uncertainty[correct_mask].mean())
    else:
        unc_sep = 0.0

    return {
        "model": model_name,
        "mse": mse,
        "mae": mae,
        "cls_acc": cls_acc,
        "unc_sep": unc_sep,
        "t1": t1,
        "t2": t2,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()
    feature_cols = CONFIG["features"]

    train_df = df[df["frame_id"] <= CONFIG["train_cutoff"]].copy()
    val_df = df[df["frame_id"] > CONFIG["train_cutoff"]].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # Sequence windows for neural models
    X_train, y_train = build_windows(train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_val, y_val = build_windows(val_df, feature_cols, CONFIG["eval_stride"], scaler)

    # Flat windows for tree models
    X_train_flat, _ = build_flat_windows(train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_val_flat, _ = build_flat_windows(val_df, feature_cols, CONFIG["eval_stride"], scaler)

    n_feat = len(feature_cols)
    n_passes = CONFIG["mc_dropout_passes"]
    results = []

    # ── 1. Plain LSTM with MSE loss ──
    print("  Training Plain LSTM...", file=sys.stderr)
    lstm = PlainLSTM(input_size=n_feat)
    lstm = train_neural(lstm, X_train, y_train, X_val, y_val, device)
    lstm_pred, lstm_var = mc_dropout_predict(lstm, X_val, device, n_passes)
    r = evaluate_model(lstm_pred, y_val, lstm_var, y_train, "plain_lstm")
    results.append(r)
    print(f"    LSTM: MSE={r['mse']:.6f} MAE={r['mae']:.6f} cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:.6f}", file=sys.stderr)

    # ── 2. Plain GRU with MSE loss ──
    print("  Training Plain GRU...", file=sys.stderr)
    gru = PlainGRU(input_size=n_feat)
    gru = train_neural(gru, X_train, y_train, X_val, y_val, device)
    gru_pred, gru_var = mc_dropout_predict(gru, X_val, device, n_passes)
    r = evaluate_model(gru_pred, y_val, gru_var, y_train, "plain_gru")
    results.append(r)
    print(f"    GRU:  MSE={r['mse']:.6f} MAE={r['mae']:.6f} cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:.6f}", file=sys.stderr)

    # ── 3. Random Forest ──
    print("  Training Random Forest...", file=sys.stderr)
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train_flat, y_train)
    rf_pred = rf.predict(X_val_flat)
    # Tree-based uncertainty: variance across trees
    rf_tree_preds = np.array([tree.predict(X_val_flat) for tree in rf.estimators_])
    rf_var = rf_tree_preds.var(axis=0)
    r = evaluate_model(rf_pred, y_val, rf_var, y_train, "random_forest")
    results.append(r)
    print(f"    RF:   MSE={r['mse']:.6f} MAE={r['mae']:.6f} cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:.6f}", file=sys.stderr)

    # ── 4. Gradient Boosting ──
    print("  Training GradientBoosting...", file=sys.stderr)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train_flat, y_train)
    gb_pred = gb.predict(X_val_flat)
    # GB uncertainty: staged predictions variance (last 50 trees)
    staged = np.array(list(gb.staged_predict(X_val_flat)))
    gb_var = staged[-50:].var(axis=0) if len(staged) >= 50 else staged.var(axis=0)
    r = evaluate_model(gb_pred, y_val, gb_var, y_train, "gradient_boosting")
    results.append(r)
    print(f"    GB:   MSE={r['mse']:.6f} MAE={r['mae']:.6f} cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:.6f}", file=sys.stderr)

    # ── 5. XGBoost ──
    if HAS_XGB:
        print("  Training XGBoost...", file=sys.stderr)
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, random_state=42, verbosity=0)
        xgb_model.fit(X_train_flat, y_train)
        xgb_pred = xgb_model.predict(X_val_flat)
        # XGB uncertainty: prediction from subsets of trees
        xgb_preds_staged = []
        for ntree in range(max(1, 150), 201):
            xgb_preds_staged.append(xgb_model.predict(X_val_flat, iteration_range=(0, ntree)))
        xgb_var = np.array(xgb_preds_staged).var(axis=0)
        r = evaluate_model(xgb_pred, y_val, xgb_var, y_train, "xgboost")
        results.append(r)
        print(f"    XGB:  MSE={r['mse']:.6f} MAE={r['mae']:.6f} cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:.6f}", file=sys.stderr)

    # ── 6. LSTM + RF Ensemble ──
    print("  Computing LSTM+RF Ensemble...", file=sys.stderr)
    ensemble_pred = 0.5 * lstm_pred + 0.5 * rf_pred
    ensemble_var = 0.5 * lstm_var + 0.5 * rf_var
    r = evaluate_model(ensemble_pred, y_val, ensemble_var, y_train, "lstm_rf_ensemble")
    results.append(r)
    print(f"    ENS:  MSE={r['mse']:.6f} MAE={r['mae']:.6f} cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:.6f}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s", file=sys.stderr)

    # Print comparison table
    print("\n  === NON-EVIDENTIAL COMPARISON ===", file=sys.stderr)
    print(f"  {'Model':<20} {'MSE':>10} {'MAE':>10} {'cls_acc':>10} {'unc_sep':>12}", file=sys.stderr)
    print(f"  {'-'*62}", file=sys.stderr)
    for r in results:
        print(f"  {r['model']:<20} {r['mse']:>10.6f} {r['mae']:>10.6f} {r['cls_acc']:>10.3f} {r['unc_sep']:>12.6f}", file=sys.stderr)

    # Output results line for each model
    for r in results:
        config = {"model": r["model"], "t1_pct": CONFIG["t1_percentile"],
                  "t2_pct": CONFIG["t2_percentile"], "n_features": len(feature_cols)}
        print(f"RESULT\tmodel={r['model']}\tmse={r['mse']:.6f}\tmae={r['mae']:.6f}\t"
              f"unc_separation={r['unc_sep']:.6f}\tcls_acc={r['cls_acc']:.6f}\t"
              f"config={json.dumps(config)}")


if __name__ == "__main__":
    main()
