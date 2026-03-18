#!/usr/bin/env python3
"""Fair non-evidential comparison: dedicated hyperparameter sweep for each model.

Each model gets 15-20 experiments optimizing cls_acc on the same miss_rate target.
"""

import json
import sys
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from features import TOP_35_FEATURES

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "yolox_features.csv"
DETS_CSV = SCRIPT_DIR / "yolox_detections.csv"
RESULTS_TSV = SCRIPT_DIR / "evidential_results.tsv"

FEATURES = TOP_35_FEATURES

W, H, SW = 30, 30, 6
N_STEPS = H // SW
TRAIN_STRIDE, EVAL_STRIDE = 10, 30
TRAIN_CUTOFF = 14400


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


def build_windows(df, stride, scaler):
    """Build multi-step windows: targets are per-sub-window means."""
    total_need = W + H
    X_list, y_list = [], []
    for _, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue
        feat_vals = scaler.transform(sdf[FEATURES].values)
        target_vals = sdf["miss_rate"].values
        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t: t + W])
            # Multi-step targets
            targets = np.zeros(N_STEPS, dtype=np.float32)
            for step in range(N_STEPS):
                s = t + W + step * SW
                targets[step] = target_vals[s:s + SW].mean()
            y_list.append(targets)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_flat_windows(df, stride, scaler):
    """Flat windows for tree models — single target (mean over horizon)."""
    total_need = W + H
    X_list, y_list = [], []
    for _, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue
        feat_vals = scaler.transform(sdf[FEATURES].values)
        target_vals = sdf["miss_rate"].values
        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t: t + W].flatten())
            y_list.append(target_vals[t + W: t + W + H].mean())
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def evaluate(y_pred, y_val, uncertainty, y_train, t1_pct, t2_pct):
    """Evaluate predictions: MSE, MAE, cls_acc, unc_sep."""
    mse = float(np.mean((y_pred - y_val) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_val)))
    t1 = float(np.percentile(y_train, t1_pct))
    t2 = float(np.percentile(y_train, t2_pct))

    pred_class = np.ones(len(y_pred), dtype=int)  # default moderate
    pred_class[y_pred < t1] = 0
    pred_class[y_pred >= t2] = 2

    true_class = np.ones(len(y_val), dtype=int)
    true_class[y_val < t1] = 0
    true_class[y_val >= t2] = 2

    cls_acc = float(np.mean(pred_class == true_class))

    correct = (pred_class == true_class)
    if correct.sum() > 0 and (~correct).sum() > 0:
        unc_sep = float(uncertainty[~correct].mean() - uncertainty[correct].mean())
    else:
        unc_sep = 0.0

    return {"mse": mse, "mae": mae, "cls_acc": cls_acc, "unc_sep": unc_sep}


# ═══════════════════════════════════════════════════════════════════
# NEURAL MODELS (multi-step output for fair comparison)
# ═══════════════════════════════════════════════════════════════════

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden, n_layers, dropout, n_steps):
        super().__init__()
        self.n_steps = n_steps
        drop = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_steps),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # (batch, n_steps)


class MultiStepGRU(nn.Module):
    def __init__(self, input_size, hidden, n_layers, dropout, n_steps):
        super().__init__()
        self.n_steps = n_steps
        drop = dropout if n_layers > 1 else 0.0
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden,
                          num_layers=n_layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_steps),
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


def train_neural(model, X_train, y_train, X_val, y_val, device, lr, batch_size,
                 scheduler_type, max_epochs=300, patience=40, grad_clip=1.0):
    model = model.to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)

    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()

        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def mc_dropout_predict_multistep(model, X, device, n_passes=10):
    """MC Dropout with multi-step output — average across steps."""
    model.train()
    x = torch.from_numpy(X).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x).cpu().numpy()  # (N, n_steps)
            preds.append(pred.mean(axis=1))  # average steps → (N,)
    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    var_pred = preds.var(axis=0)
    model.eval()
    return mean_pred, var_pred


def log_result(commit, mse, mae, unc_sep, cls_acc, n_val, status, desc):
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{mse:.6f}\t{mae:.6f}\t{unc_sep:.6f}\t{cls_acc:.6f}\t-\t{n_val}\t{status}\t{desc}\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = "fair"

    df = load_data()
    train_df = df[df["frame_id"] <= TRAIN_CUTOFF].copy()
    val_df = df[df["frame_id"] > TRAIN_CUTOFF].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES].values)

    # Multi-step windows for neural models
    X_train, y_train = build_windows(train_df, TRAIN_STRIDE, scaler)
    X_val, y_val = build_windows(val_df, EVAL_STRIDE, scaler)
    y_train_flat = y_train.mean(axis=1)
    y_val_flat = y_val.mean(axis=1)
    n_val = len(y_val)
    n_feat = len(FEATURES)

    # Flat windows for tree models
    X_train_flat, y_train_tree = build_flat_windows(train_df, TRAIN_STRIDE, scaler)
    X_val_flat, y_val_tree = build_flat_windows(val_df, EVAL_STRIDE, scaler)

    print(f"  Data: train={len(X_train)} val={n_val} feat={n_feat} steps={N_STEPS}", file=sys.stderr)

    all_results = {}

    # ════════════════════════════════════════════════════════════════
    # 1. PLAIN LSTM SWEEP (20 experiments)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60, file=sys.stderr)
    print("  PLAIN LSTM SWEEP", file=sys.stderr)
    print("="*60, file=sys.stderr)

    lstm_best = {"cls_acc": 0, "config": {}}
    lstm_configs = [
        # (lr, dropout, hidden, t1, t2, scheduler)
        (1e-3, 0.4, 128, 10, 85, "cosine"),    # NIG-matched baseline
        (1e-3, 0.3, 128, 10, 85, "cosine"),
        (1e-3, 0.2, 128, 10, 85, "cosine"),
        (5e-4, 0.4, 128, 10, 85, "cosine"),
        (2e-3, 0.4, 128, 10, 85, "cosine"),
        (1e-3, 0.4, 64, 10, 85, "cosine"),
        (1e-3, 0.4, 128, 15, 85, "cosine"),
        (1e-3, 0.4, 128, 20, 85, "cosine"),
        (1e-3, 0.4, 128, 10, 80, "cosine"),
        (1e-3, 0.4, 128, 25, 80, "cosine"),
        (1e-3, 0.4, 128, 10, 85, "plateau"),
        (1e-3, 0.3, 64, 10, 85, "cosine"),
        (2e-3, 0.3, 128, 10, 85, "cosine"),
        (1e-3, 0.4, 128, 15, 80, "cosine"),
        (5e-4, 0.3, 128, 10, 85, "plateau"),
        (1e-3, 0.2, 64, 15, 85, "cosine"),
        (2e-3, 0.4, 128, 15, 85, "cosine"),
        (1e-3, 0.3, 128, 20, 80, "cosine"),
        (1e-3, 0.4, 128, 25, 85, "cosine"),
        (5e-4, 0.4, 64, 10, 85, "cosine"),
    ]

    for i, (lr, drop, h, t1, t2, sched) in enumerate(lstm_configs):
        desc = f"fair_lstm lr={lr} drop={drop} h={h} t1={t1}/t2={t2} {sched}"
        try:
            model = MultiStepLSTM(n_feat, h, 2, drop, N_STEPS)
            model = train_neural(model, X_train, y_train, X_val, y_val, device,
                                 lr, 64, sched)
            pred, var = mc_dropout_predict_multistep(model, X_val, device, 10)
            r = evaluate(pred, y_val_flat, var, y_train_flat, t1, t2)
            log_result(commit, r["mse"], r["mae"], r["unc_sep"], r["cls_acc"], n_val, "fair_lstm", desc)
            print(f"  LSTM {i+1:2d}/20: cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:+.4f} | {desc}", file=sys.stderr)
            if r["cls_acc"] > lstm_best["cls_acc"]:
                lstm_best = {**r, "config": desc}
        except Exception as e:
            print(f"  LSTM {i+1:2d}/20: FAILED — {e}", file=sys.stderr)

    all_results["plain_lstm"] = lstm_best
    print(f"\n  >>> LSTM BEST: cls_acc={lstm_best['cls_acc']:.3f} | {lstm_best['config']}", file=sys.stderr)

    # ════════════════════════════════════════════════════════════════
    # 2. PLAIN GRU SWEEP (20 experiments)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60, file=sys.stderr)
    print("  PLAIN GRU SWEEP", file=sys.stderr)
    print("="*60, file=sys.stderr)

    gru_best = {"cls_acc": 0, "config": {}}
    gru_configs = [
        (1e-3, 0.4, 128, 10, 85, "cosine"),
        (1e-3, 0.3, 128, 10, 85, "cosine"),
        (1e-3, 0.2, 128, 10, 85, "cosine"),
        (5e-4, 0.4, 128, 10, 85, "cosine"),
        (2e-3, 0.4, 128, 10, 85, "cosine"),
        (1e-3, 0.4, 64, 10, 85, "cosine"),
        (1e-3, 0.4, 128, 15, 85, "cosine"),
        (1e-3, 0.4, 128, 20, 85, "cosine"),
        (1e-3, 0.4, 128, 10, 80, "cosine"),
        (1e-3, 0.4, 128, 25, 80, "cosine"),
        (1e-3, 0.4, 128, 10, 85, "plateau"),
        (1e-3, 0.3, 64, 10, 85, "cosine"),
        (2e-3, 0.3, 128, 10, 85, "cosine"),
        (1e-3, 0.4, 128, 15, 80, "cosine"),
        (5e-4, 0.3, 128, 10, 85, "plateau"),
        (1e-3, 0.2, 64, 15, 85, "cosine"),
        (2e-3, 0.4, 128, 15, 85, "cosine"),
        (1e-3, 0.3, 128, 20, 80, "cosine"),
        (1e-3, 0.4, 128, 25, 85, "cosine"),
        (5e-4, 0.4, 64, 10, 85, "cosine"),
    ]

    for i, (lr, drop, h, t1, t2, sched) in enumerate(gru_configs):
        desc = f"fair_gru lr={lr} drop={drop} h={h} t1={t1}/t2={t2} {sched}"
        try:
            model = MultiStepGRU(n_feat, h, 2, drop, N_STEPS)
            model = train_neural(model, X_train, y_train, X_val, y_val, device,
                                 lr, 64, sched)
            pred, var = mc_dropout_predict_multistep(model, X_val, device, 10)
            r = evaluate(pred, y_val_flat, var, y_train_flat, t1, t2)
            log_result(commit, r["mse"], r["mae"], r["unc_sep"], r["cls_acc"], n_val, "fair_gru", desc)
            print(f"  GRU  {i+1:2d}/20: cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:+.4f} | {desc}", file=sys.stderr)
            if r["cls_acc"] > gru_best["cls_acc"]:
                gru_best = {**r, "config": desc}
        except Exception as e:
            print(f"  GRU  {i+1:2d}/20: FAILED — {e}", file=sys.stderr)

    all_results["plain_gru"] = gru_best
    print(f"\n  >>> GRU BEST: cls_acc={gru_best['cls_acc']:.3f} | {gru_best['config']}", file=sys.stderr)

    # ════════════════════════════════════════════════════════════════
    # 3. RANDOM FOREST SWEEP (15 experiments)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60, file=sys.stderr)
    print("  RANDOM FOREST SWEEP", file=sys.stderr)
    print("="*60, file=sys.stderr)

    rf_best = {"cls_acc": 0, "config": {}}
    rf_configs = [
        # (n_est, max_depth, min_leaf, t1, t2)
        (300, 15, 5, 10, 85),
        (500, 15, 5, 10, 85),
        (1000, 15, 5, 10, 85),
        (300, 20, 3, 10, 85),
        (300, 30, 2, 10, 85),
        (300, 10, 10, 10, 85),
        (500, 20, 3, 10, 85),
        (300, 15, 5, 15, 85),
        (300, 15, 5, 20, 85),
        (300, 15, 5, 10, 80),
        (300, 15, 5, 25, 80),
        (500, 25, 2, 10, 85),
        (1000, 20, 3, 10, 85),
        (300, None, 3, 10, 85),
        (500, 15, 5, 15, 80),
    ]

    rf_best_model = None
    for i, (n_est, max_d, min_leaf, t1, t2) in enumerate(rf_configs):
        desc = f"fair_rf n={n_est} d={max_d} leaf={min_leaf} t1={t1}/t2={t2}"
        try:
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                        min_samples_leaf=min_leaf, n_jobs=-1, random_state=42)
            rf.fit(X_train_flat, y_train_tree)
            pred = rf.predict(X_val_flat)
            tree_preds = np.array([tree.predict(X_val_flat) for tree in rf.estimators_])
            var = tree_preds.var(axis=0)
            r = evaluate(pred, y_val_tree, var, y_train_tree, t1, t2)
            log_result(commit, r["mse"], r["mae"], r["unc_sep"], r["cls_acc"], n_val, "fair_rf", desc)
            print(f"  RF   {i+1:2d}/15: cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:+.4f} | {desc}", file=sys.stderr)
            if r["cls_acc"] > rf_best["cls_acc"]:
                rf_best = {**r, "config": desc}
                rf_best_model = rf
        except Exception as e:
            print(f"  RF   {i+1:2d}/15: FAILED — {e}", file=sys.stderr)

    all_results["random_forest"] = rf_best
    print(f"\n  >>> RF BEST: cls_acc={rf_best['cls_acc']:.3f} | {rf_best['config']}", file=sys.stderr)

    # ════════════════════════════════════════════════════════════════
    # 4. XGBOOST SWEEP (15 experiments)
    # ════════════════════════════════════════════════════════════════
    if HAS_XGB:
        print("\n" + "="*60, file=sys.stderr)
        print("  XGBOOST SWEEP", file=sys.stderr)
        print("="*60, file=sys.stderr)

        xgb_best = {"cls_acc": 0, "config": {}}
        xgb_configs = [
            # (n_est, max_depth, lr, subsample, t1, t2)
            (300, 5, 0.05, 0.8, 10, 85),
            (500, 4, 0.05, 0.8, 10, 85),
            (300, 6, 0.05, 0.8, 10, 85),
            (300, 8, 0.05, 0.8, 10, 85),
            (300, 5, 0.01, 0.8, 10, 85),
            (300, 5, 0.1, 0.8, 10, 85),
            (500, 3, 0.02, 0.9, 10, 85),
            (300, 5, 0.03, 0.8, 10, 85),
            (300, 5, 0.05, 0.8, 15, 85),
            (300, 5, 0.05, 0.8, 20, 85),
            (300, 5, 0.05, 0.8, 10, 80),
            (300, 5, 0.05, 0.8, 25, 80),
            (500, 5, 0.03, 0.9, 10, 85),
            (1000, 4, 0.01, 0.8, 10, 85),
            (300, 5, 0.05, 0.7, 15, 80),
        ]

        for i, (n_est, max_d, lr, sub, t1, t2) in enumerate(xgb_configs):
            desc = f"fair_xgb n={n_est} d={max_d} lr={lr} sub={sub} t1={t1}/t2={t2}"
            try:
                m = xgb.XGBRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                                     subsample=sub, random_state=42, verbosity=0)
                m.fit(X_train_flat, y_train_tree)
                pred = m.predict(X_val_flat)
                preds_s = [m.predict(X_val_flat, iteration_range=(0, j))
                           for j in range(max(1, n_est - 50), n_est + 1)]
                var = np.array(preds_s).var(axis=0)
                r = evaluate(pred, y_val_tree, var, y_train_tree, t1, t2)
                log_result(commit, r["mse"], r["mae"], r["unc_sep"], r["cls_acc"], n_val, "fair_xgb", desc)
                print(f"  XGB  {i+1:2d}/15: cls_acc={r['cls_acc']:.3f} unc_sep={r['unc_sep']:+.4f} | {desc}", file=sys.stderr)
                if r["cls_acc"] > xgb_best["cls_acc"]:
                    xgb_best = {**r, "config": desc}
            except Exception as e:
                print(f"  XGB  {i+1:2d}/15: FAILED — {e}", file=sys.stderr)

        all_results["xgboost"] = xgb_best
        print(f"\n  >>> XGB BEST: cls_acc={xgb_best['cls_acc']:.3f} | {xgb_best['config']}", file=sys.stderr)

    # ════════════════════════════════════════════════════════════════
    # 5. LSTM+RF ENSEMBLE (use best LSTM + best RF)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60, file=sys.stderr)
    print("  LSTM+RF ENSEMBLE (best of each)", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # Re-train best LSTM config
    best_lstm_cfg = lstm_configs[0]  # default
    best_lstm_acc = 0
    for cfg in lstm_configs:
        lr, drop, h, t1, t2, sched = cfg
        # Find the config that matched lstm_best
        test_desc = f"fair_lstm lr={lr} drop={drop} h={h} t1={t1}/t2={t2} {sched}"
        if test_desc == lstm_best.get("config", ""):
            best_lstm_cfg = cfg
            break

    lr, drop, h, t1_l, t2_l, sched = best_lstm_cfg
    model_lstm = MultiStepLSTM(n_feat, h, 2, drop, N_STEPS)
    model_lstm = train_neural(model_lstm, X_train, y_train, X_val, y_val, device, lr, 64, sched)
    lstm_pred, lstm_var = mc_dropout_predict_multistep(model_lstm, X_val, device, 10)

    # Best RF predictions
    if rf_best_model is not None:
        rf_pred = rf_best_model.predict(X_val_flat)
        rf_tree_preds = np.array([t.predict(X_val_flat) for t in rf_best_model.estimators_])
        rf_var = rf_tree_preds.var(axis=0)
    else:
        rf_pred = np.zeros(n_val)
        rf_var = np.zeros(n_val)

    # Try different blend weights and thresholds
    ens_best = {"cls_acc": 0, "config": {}}
    for w_lstm in [0.3, 0.5, 0.7]:
        for t1, t2 in [(10, 85), (15, 85), (10, 80), (15, 80), (20, 85)]:
            ens_pred = w_lstm * lstm_pred + (1 - w_lstm) * rf_pred
            ens_var = w_lstm**2 * lstm_var + (1 - w_lstm)**2 * rf_var  # Var(wA+(1-w)B)
            r = evaluate(ens_pred, y_val_tree, ens_var, y_train_tree, t1, t2)
            desc = f"fair_ens w_lstm={w_lstm} t1={t1}/t2={t2}"
            log_result(commit, r["mse"], r["mae"], r["unc_sep"], r["cls_acc"], n_val, "fair_ens", desc)
            print(f"  ENS: cls_acc={r['cls_acc']:.3f} | {desc}", file=sys.stderr)
            if r["cls_acc"] > ens_best["cls_acc"]:
                ens_best = {**r, "config": desc}

    all_results["lstm_rf_ensemble"] = ens_best
    print(f"\n  >>> ENS BEST: cls_acc={ens_best['cls_acc']:.3f} | {ens_best['config']}", file=sys.stderr)

    # ════════════════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"  FINAL COMPARISON — EACH MODEL FULLY TUNED ({elapsed:.0f}s total)", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    # Load NIG results from results TSV instead of hardcoding
    nig_row = None
    if RESULTS_TSV.exists():
        import csv
        with open(RESULTS_TSV) as f:
            rows = [r for r in csv.reader(f, delimiter='\t') if len(r) >= 8 and r[7] == 'evidential_nig']
            if rows:
                row = rows[-1]  # latest run
                nig_row = {"cls_acc": float(row[4]), "unc_sep": float(row[3]),
                           "mse": float(row[1]), "mae": float(row[2]),
                           "config": row[8] if len(row) > 8 else ""}

    print(f"  {'Model':<22} {'cls_acc':>10} {'unc_sep':>12} {'MSE':>10} {'MAE':>10}   Config", file=sys.stderr)
    print(f"  {'-'*100}", file=sys.stderr)
    if nig_row:
        print(f"  {'EVIDENTIAL NIG':<22} {nig_row['cls_acc']:>10.3f} {nig_row['unc_sep']:>+12.6f} {nig_row['mse']:>10.6f} {nig_row['mae']:>10.6f}   {nig_row['config']}", file=sys.stderr)
    else:
        print(f"  {'EVIDENTIAL NIG':<22} {'(no results — run train_reg.py first)':>45}", file=sys.stderr)
    for name, r in all_results.items():
        if r["cls_acc"] > 0:
            print(f"  {name:<22} {r['cls_acc']:>10.3f} {r['unc_sep']:>+12.6f} {r['mse']:>10.6f} {r['mae']:>10.6f}   {r.get('config','')}", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)

    # Also print to stdout for capture
    print(f"\nFINAL_TABLE")
    if nig_row:
        print(f"EVIDENTIAL_NIG\t{nig_row['cls_acc']:.6f}\t{nig_row['unc_sep']:.6f}\t{nig_row['mse']:.6f}\t{nig_row['mae']:.6f}")
    for name, r in all_results.items():
        if r["cls_acc"] > 0:
            print(f"{name}\t{r['cls_acc']:.6f}\t{r['unc_sep']:.6f}\t{r['mse']:.6f}\t{r['mae']:.6f}")


if __name__ == "__main__":
    main()
