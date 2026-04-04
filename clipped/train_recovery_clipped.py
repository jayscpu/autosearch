#!/usr/bin/env python3
"""
Multi-output LSTM predicting miss_rate_nano + recovery deltas.

Targets:
  1. miss_rate_nano           — baseline miss rate
  2. recovery_nano_to_small   — miss_rate_nano - miss_rate_small  (how much upgrading to small helps)
  3. recovery_small_to_medium — miss_rate_small - miss_rate_medium (how much upgrading to medium helps)

Same architecture as train_3target.py (PlainLSTM, h256, L4, SPATIAL_65, w30/h30/sw6).

Outputs predictions_recovery.csv with columns:
  frame_idx, pred_nano, pred_recovery_ns, pred_recovery_sm,
  true_nano, true_recovery_ns, true_recovery_sm, intersection
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

# ── Config (matching plainlstm_4cam / train_3target.py) ──
CONFIG = {
    "features": SPATIAL_65,
    "targets": ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"],
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
    "seeds": [42],
}

N_TARGETS = len(CONFIG["targets"])


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════

def load_data():
    print("  Loading pod_features_all.csv ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)
    df["miss_rate_nano"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    df["miss_rate_small"] = df["fn_small"] / df["x_count"].clip(lower=1)
    df["miss_rate_medium"] = df["fn_medium"] / df["x_count"].clip(lower=1)
    # Recovery deltas (positive = upgrading helps)
    df["recovery_nano_to_small"] = df["miss_rate_nano"] - df["miss_rate_small"]
    df["recovery_small_to_medium"] = (df["miss_rate_small"] - df["miss_rate_medium"]).clip(lower=0)
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


def build_windows(df, feature_cols, target_cols, stride, scaler):
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
        target_vals = sdf[target_cols].values  # (n, N_TARGETS)

        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t:t + W])

            targets = np.zeros((n_steps, len(target_cols)), dtype=np.float32)
            for step in range(n_steps):
                start = t + W + step * S
                end = start + S
                targets[step] = target_vals[start:end].mean(axis=0)

            y_list.append(targets)
            seq_labels.append(seq_name)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, seq_labels


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class MultiOutputLSTM(nn.Module):
    """PlainLSTM with N_TARGETS outputs per step (autoregressive decoder)."""

    def __init__(self, input_size, n_steps, n_targets):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0
        self.n_steps = n_steps
        self.n_targets = n_targets

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTMCell(input_size=n_targets, hidden_size=hidden)
        self.step_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, n_targets),
        )

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_dec = h_n[-1]
        c_dec = c_n[-1]

        preds = []
        dec_input = torch.zeros(x.size(0), self.n_targets, device=x.device)
        for _ in range(self.n_steps):
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            step_pred = self.step_head(h_dec)
            preds.append(step_pred)
            dec_input = step_pred.detach()

        return torch.stack(preds, dim=1)


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, n_feat, n_steps, device):
    model = MultiOutputLSTM(input_size=n_feat, n_steps=n_steps,
                            n_targets=N_TARGETS).to(device)

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
            print(f"    epoch {epoch:3d}: val_loss={val_loss:.6f} "
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
    target_cols = CONFIG["targets"]

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

    # Build windows
    X_train, y_train, _ = build_windows(
        train_df, feature_cols, target_cols, CONFIG["train_stride"], scaler)
    X_earlystop, y_earlystop, _ = build_windows(
        earlystop_df, feature_cols, target_cols, CONFIG["eval_stride"], scaler)
    X_within, y_within, seq_within = build_windows(
        within_val_df, feature_cols, target_cols, CONFIG["eval_stride"], scaler)
    X_cross, y_cross, seq_cross = build_windows(
        cross_val_df, feature_cols, target_cols, CONFIG["eval_stride"], scaler)

    n_feat = len(feature_cols)
    print(f"  Windows: train={len(y_train)}, earlystop={len(y_earlystop)}, "
          f"within={len(y_within)}, cross={len(y_cross)}", flush=True)
    print(f"  n_steps={n_steps}, n_targets={N_TARGETS}, n_features={n_feat}",
          flush=True)

    # Combine within + cross for evaluation
    X_eval = np.concatenate([X_within, X_cross], axis=0)
    y_eval = np.concatenate([y_within, y_cross], axis=0)
    seq_eval = seq_within + seq_cross

    y_eval_mean = y_eval.mean(axis=1)  # (N, N_TARGETS)

    # ── Training ──
    print(f"\n  Training (seed 42) ...", flush=True)
    all_preds = []
    for seed in CONFIG["seeds"]:
        print(f"\n  ── Seed {seed} ──", flush=True)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = train_model(X_train, y_train, X_earlystop, y_earlystop,
                            n_feat, n_steps, device)
        with torch.no_grad():
            eval_preds = []
            for start in range(0, len(X_eval), 512):
                xb = torch.from_numpy(X_eval[start:start + 512]).to(device)
                eval_preds.append(model(xb).cpu().numpy())
            p = np.concatenate(eval_preds, axis=0)
        all_preds.append(p)

    pred_mean = np.mean(all_preds, axis=0).mean(axis=1)  # (N, N_TARGETS)

    # ── Per-target MSE ──
    print(f"\n  Per-target MSE (eval, n={len(y_eval_mean)}):", flush=True)
    for i, name in enumerate(target_cols):
        mse = float(np.mean((pred_mean[:, i] - y_eval_mean[:, i]) ** 2))
        mae = float(np.mean(np.abs(pred_mean[:, i] - y_eval_mean[:, i])))
        corr = float(np.corrcoef(pred_mean[:, i], y_eval_mean[:, i])[0, 1])
        print(f"    {name:28s}: MSE={mse:.6f}  MAE={mae:.4f}  r={corr:.4f}",
              flush=True)

    # ── Save predictions CSV ──
    rows = []
    for i in range(len(y_eval_mean)):
        rows.append({
            "frame_idx": i,
            "pred_nano": round(float(pred_mean[i, 0]), 6),
            "pred_recovery_ns": round(float(pred_mean[i, 1]), 6),
            "pred_recovery_sm": round(float(pred_mean[i, 2]), 6),
            "true_nano": round(float(y_eval_mean[i, 0]), 6),
            "true_recovery_ns": round(float(y_eval_mean[i, 1]), 6),
            "true_recovery_sm": round(float(y_eval_mean[i, 2]), 6),
            "intersection": seq_eval[i],
        })

    out_path = SCRIPT_DIR / "predictions_recovery_clipped.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  Saved {out_path} ({len(rows)} rows)", flush=True)

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
