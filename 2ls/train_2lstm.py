#!/usr/bin/env python3
"""
Two separate single-output LSTMs for recovery prediction.

LSTM-A: predicts recovery_nano_to_small only
LSTM-B: predicts recovery_small_to_medium only

Same architecture as train_recovery.py (PlainLSTM h256 L4 d0.4, SPATIAL_65,
w30/h30/sw6) but each model has its own encoder and decoder, eliminating
interference between targets.

Outputs 2ls/predictions_2lstm.csv.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "second_autosearch" / "data"
FEATURES_CSV = DATA_DIR / "pod_features_all.csv"

sys.path.insert(0, str(PROJECT_DIR / "second_autosearch" / "src"))
from pod_features import SPATIAL_65

# ── Config (matching train_recovery.py) ──
CONFIG = {
    "features": SPATIAL_65,
    "window": 30,
    "horizon": 30,
    "sub_window": 6,
    "train_stride": 15,
    "eval_stride": 30,
    "warmup_frames": 400,
    "train_intersections": [
        "Bellevue_150th_Eastgate",
        "Bellevue_150th_Newport",
        "Bellevue_Bellevue_NE8th",
        "Bellevue_116th_NE12th",
    ],
    "test_intersection": "Bellevue_150th_SE38th",
    "train_fraction": 0.50,
    "earlystop_fraction": 0.60,
    "hidden_size": 256,
    "n_layers": 4,
    "dropout": 0.4,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 300,
    "patience": 40,
    "grad_clip": 1.0,
    "seed": 42,
}


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════

def load_data():
    print("  Loading pod_features_all.csv ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)
    df["miss_rate_nano"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    df["miss_rate_small"] = df["fn_small"] / df["x_count"].clip(lower=1)
    df["miss_rate_medium"] = df["fn_medium"] / df["x_count"].clip(lower=1)
    df["recovery_nano_to_small"] = df["miss_rate_nano"] - df["miss_rate_small"]
    df["recovery_small_to_medium"] = df["miss_rate_small"] - df["miss_rate_medium"]
    return df


def split_data(df):
    train_dfs, earlystop_dfs, within_val_dfs = [], [], []
    for int_name in CONFIG["train_intersections"]:
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
    cross_val_df = df[df["sequence"] == CONFIG["test_intersection"]].copy()
    return train_df, earlystop_df, within_val_df, cross_val_df


def build_windows(df, feature_cols, target_col, stride, scaler):
    """Build windows for a single target column."""
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S
    total_need = W + H

    X_list, y_list, seq_labels = [], [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = CONFIG["warmup_frames"]

    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        seq_name = sdf["sequence"].iloc[0]
        feat_vals = scaler.transform(sdf[feature_cols].values)
        target_vals = sdf[target_col].values

        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t:t + W])

            targets = np.zeros(n_steps, dtype=np.float32)
            for step in range(n_steps):
                start = t + W + step * S
                end = start + S
                targets[step] = target_vals[start:end].mean()

            y_list.append(targets)
            seq_labels.append(seq_name)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, seq_labels


def build_windows_multi(df, feature_cols, target_cols, stride, scaler):
    """Build windows for multiple targets (for ground truth export)."""
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S
    total_need = W + H

    y_list, seq_labels = [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = CONFIG["warmup_frames"]

    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        seq_name = sdf["sequence"].iloc[0]
        target_vals = sdf[target_cols].values

        for t in range(0, n - total_need + 1, stride):
            targets = np.zeros((n_steps, len(target_cols)), dtype=np.float32)
            for step in range(n_steps):
                start = t + W + step * S
                end = start + S
                targets[step] = target_vals[start:end].mean(axis=0)

            y_list.append(targets)
            seq_labels.append(seq_name)

    y = np.array(y_list, dtype=np.float32)
    return y, seq_labels


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class SingleOutputLSTM(nn.Module):
    """PlainLSTM with 1 output per step (autoregressive decoder)."""

    def __init__(self, input_size, n_steps):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0
        self.n_steps = n_steps

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTMCell(input_size=1, hidden_size=hidden)
        self.step_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_dec = h_n[-1]
        c_dec = c_n[-1]

        preds = []
        dec_input = torch.zeros(x.size(0), 1, device=x.device)
        for _ in range(self.n_steps):
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            step_pred = self.step_head(h_dec)
            preds.append(step_pred)
            dec_input = step_pred.detach()

        return torch.stack(preds, dim=1).squeeze(-1)  # (batch, n_steps)


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, n_feat, n_steps, device, label):
    model = SingleOutputLSTM(input_size=n_feat, n_steps=n_steps).to(device)

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

        if epoch % 10 == 0 or patience_ctr >= CONFIG["patience"]:
            print(f"    [{label}] epoch {epoch:3d}: val_loss={val_loss:.6f} "
                  f"best={best_val_loss:.6f} patience={patience_ctr}",
                  flush=True)

        if patience_ctr >= CONFIG["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
    feature_cols = CONFIG["features"]

    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S

    train_df, earlystop_df, within_val_df, cross_val_df = split_data(df)
    print(f"  Train: {len(train_df)}, EarlyStop: {len(earlystop_df)}, "
          f"Within: {len(within_val_df)}, Cross: {len(cross_val_df)}", flush=True)

    # Scaler from training data only (warmup-skipped)
    warmup = CONFIG["warmup_frames"]
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[warmup:])
    scaler_df = pd.concat(kept, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(scaler_df[feature_cols].values)

    n_feat = len(feature_cols)

    # Ground truth for all targets (for CSV export)
    truth_targets = ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"]
    eval_dfs = [within_val_df, cross_val_df]
    eval_combined = pd.concat(eval_dfs, ignore_index=True)

    # ── Train LSTM-A: recovery_nano_to_small ──
    target_a = "recovery_nano_to_small"
    print(f"\n  === LSTM-A: {target_a} ===", flush=True)

    seed = CONFIG["seed"]
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train_a, y_train_a, _ = build_windows(
        train_df, feature_cols, target_a, CONFIG["train_stride"], scaler)
    X_es_a, y_es_a, _ = build_windows(
        earlystop_df, feature_cols, target_a, CONFIG["eval_stride"], scaler)
    X_within_a, y_within_a, seq_within_a = build_windows(
        within_val_df, feature_cols, target_a, CONFIG["eval_stride"], scaler)
    X_cross_a, y_cross_a, seq_cross_a = build_windows(
        cross_val_df, feature_cols, target_a, CONFIG["eval_stride"], scaler)

    X_eval_a = np.concatenate([X_within_a, X_cross_a], axis=0)
    y_eval_a = np.concatenate([y_within_a, y_cross_a], axis=0)
    seq_eval = seq_within_a + seq_cross_a

    print(f"  Windows: train={len(y_train_a)}, earlystop={len(y_es_a)}, "
          f"eval={len(y_eval_a)}", flush=True)

    model_a = train_model(X_train_a, y_train_a, X_es_a, y_es_a,
                          n_feat, n_steps, device, "LSTM-A")

    with torch.no_grad():
        preds_a = []
        for start in range(0, len(X_eval_a), 512):
            xb = torch.from_numpy(X_eval_a[start:start + 512]).to(device)
            preds_a.append(model_a(xb).cpu().numpy())
    pred_a = np.concatenate(preds_a, axis=0).mean(axis=1)  # (N,)
    true_a = y_eval_a.mean(axis=1)  # (N,)

    mse_a = float(np.mean((pred_a - true_a) ** 2))
    mae_a = float(np.mean(np.abs(pred_a - true_a)))
    corr_a = float(np.corrcoef(pred_a, true_a)[0, 1])
    print(f"\n  LSTM-A results: MSE={mse_a:.6f}  MAE={mae_a:.4f}  r={corr_a:.4f}",
          flush=True)

    t_a = time.time() - t0
    print(f"  LSTM-A time: {t_a:.1f}s", flush=True)

    # ── Train LSTM-B: recovery_small_to_medium ──
    target_b = "recovery_small_to_medium"
    print(f"\n  === LSTM-B: {target_b} ===", flush=True)

    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train_b, y_train_b, _ = build_windows(
        train_df, feature_cols, target_b, CONFIG["train_stride"], scaler)
    X_es_b, y_es_b, _ = build_windows(
        earlystop_df, feature_cols, target_b, CONFIG["eval_stride"], scaler)
    X_within_b, y_within_b, _ = build_windows(
        within_val_df, feature_cols, target_b, CONFIG["eval_stride"], scaler)
    X_cross_b, y_cross_b, _ = build_windows(
        cross_val_df, feature_cols, target_b, CONFIG["eval_stride"], scaler)

    X_eval_b = np.concatenate([X_within_b, X_cross_b], axis=0)
    y_eval_b = np.concatenate([y_within_b, y_cross_b], axis=0)

    print(f"  Windows: train={len(y_train_b)}, earlystop={len(y_es_b)}, "
          f"eval={len(y_eval_b)}", flush=True)

    model_b = train_model(X_train_b, y_train_b, X_es_b, y_es_b,
                          n_feat, n_steps, device, "LSTM-B")

    with torch.no_grad():
        preds_b = []
        for start in range(0, len(X_eval_b), 512):
            xb = torch.from_numpy(X_eval_b[start:start + 512]).to(device)
            preds_b.append(model_b(xb).cpu().numpy())
    pred_b = np.concatenate(preds_b, axis=0).mean(axis=1)
    true_b = y_eval_b.mean(axis=1)

    mse_b = float(np.mean((pred_b - true_b) ** 2))
    mae_b = float(np.mean(np.abs(pred_b - true_b)))
    corr_b = float(np.corrcoef(pred_b, true_b)[0, 1])
    print(f"\n  LSTM-B results: MSE={mse_b:.6f}  MAE={mae_b:.4f}  r={corr_b:.4f}",
          flush=True)

    # ── Get true_nano for each eval window (needed by cascading_direct.py) ──
    y_truth, _ = build_windows_multi(
        pd.concat([within_val_df, cross_val_df], ignore_index=True),
        feature_cols, truth_targets, CONFIG["eval_stride"], scaler)
    y_truth_mean = y_truth.mean(axis=1)  # (N, 3)
    true_nano = y_truth_mean[:, 0]

    # ── Summary comparison ──
    print(f"\n{'='*70}")
    print(f"  CORRELATION COMPARISON: Shared vs Separate LSTMs")
    print(f"{'='*70}")
    print(f"  {'Target':30s} | {'Shared (r)':>10s} | {'Separate (r)':>12s} | {'Delta':>8s}")
    print(f"  {'-'*64}")
    shared_ns, shared_sm = 0.5626, 0.2092
    delta_ns = corr_a - shared_ns
    delta_sm = corr_b - shared_sm
    print(f"  {'recovery_nano_to_small':30s} | {shared_ns:>10.4f} | {corr_a:>12.4f} | {delta_ns:>+8.4f}")
    print(f"  {'recovery_small_to_medium':30s} | {shared_sm:>10.4f} | {corr_b:>12.4f} | {delta_sm:>+8.4f}")
    print(f"{'='*70}")

    # ── Save predictions CSV ──
    rows = []
    for i in range(len(pred_a)):
        rows.append({
            "frame_idx": i,
            "pred_recovery_ns": round(float(pred_a[i]), 6),
            "pred_recovery_sm": round(float(pred_b[i]), 6),
            "true_nano": round(float(true_nano[i]), 6),
            "true_recovery_ns": round(float(true_a[i]), 6),
            "true_recovery_sm": round(float(true_b[i]), 6),
            "intersection": seq_eval[i],
        })

    out_path = SCRIPT_DIR / "predictions_2lstm.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  Saved {out_path} ({len(rows)} rows)", flush=True)

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
