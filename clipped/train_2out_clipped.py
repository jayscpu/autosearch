#!/usr/bin/env python3
"""
2-output LSTM predicting recovery_ns + recovery_sm (no miss_rate_nano).
Clips recovery_small_to_medium targets at 0.
Accepts --hidden and --layers CLI args for architecture sweeps.
"""

import argparse
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "second_autosearch" / "data"
FEATURES_CSV = DATA_DIR / "pod_features_all.csv"

sys.path.insert(0, str(PROJECT_DIR / "second_autosearch" / "src"))
from pod_features import SPATIAL_65

BASE_CONFIG = {
    "features": SPATIAL_65,
    "targets": ["recovery_nano_to_small", "recovery_small_to_medium"],
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
    "dropout": 0.4,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 300,
    "patience": 40,
    "grad_clip": 1.0,
    "seed": 42,
}


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["miss_rate_nano"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    df["miss_rate_small"] = df["fn_small"] / df["x_count"].clip(lower=1)
    df["miss_rate_medium"] = df["fn_medium"] / df["x_count"].clip(lower=1)
    df["recovery_nano_to_small"] = df["miss_rate_nano"] - df["miss_rate_small"]
    df["recovery_small_to_medium"] = (df["miss_rate_small"] - df["miss_rate_medium"]).clip(lower=0)
    return df


def split_data(df):
    train_dfs, earlystop_dfs, within_val_dfs = [], [], []
    for int_name in BASE_CONFIG["train_intersections"]:
        sub = df[df["sequence"] == int_name].sort_values("frame_id")
        n = len(sub)
        train_end = int(n * BASE_CONFIG["train_fraction"])
        earlystop_end = int(n * BASE_CONFIG["earlystop_fraction"])
        train_dfs.append(sub.iloc[:train_end])
        earlystop_dfs.append(sub.iloc[train_end:earlystop_end])
        within_val_dfs.append(sub.iloc[earlystop_end:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    earlystop_df = pd.concat(earlystop_dfs, ignore_index=True)
    within_val_df = pd.concat(within_val_dfs, ignore_index=True)
    cross_val_df = df[df["sequence"] == BASE_CONFIG["test_intersection"]].copy()
    return train_df, earlystop_df, within_val_df, cross_val_df


def build_windows(df, feature_cols, target_cols, stride, scaler):
    W = BASE_CONFIG["window"]
    H = BASE_CONFIG["horizon"]
    S = BASE_CONFIG["sub_window"]
    n_steps = H // S
    total_need = W + H

    X_list, y_list, seq_labels = [], [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = BASE_CONFIG["warmup_frames"]

    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        seq_name = sdf["sequence"].iloc[0]
        feat_vals = scaler.transform(sdf[feature_cols].values)
        target_vals = sdf[target_cols].values

        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t:t + W])
            targets = np.zeros((n_steps, len(target_cols)), dtype=np.float32)
            for step in range(n_steps):
                start = t + W + step * S
                end = start + S
                targets[step] = target_vals[start:end].mean(axis=0)
            y_list.append(targets)
            seq_labels.append(seq_name)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), seq_labels


class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, n_steps, n_targets, hidden_size, n_layers, dropout):
        super().__init__()
        self.n_steps = n_steps
        self.n_targets = n_targets
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.decoder = nn.LSTMCell(input_size=n_targets, hidden_size=hidden_size)
        self.step_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_targets),
        )

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_dec, c_dec = h_n[-1], c_n[-1]
        preds = []
        dec_input = torch.zeros(x.size(0), self.n_targets, device=x.device)
        for _ in range(self.n_steps):
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            step_pred = self.step_head(h_dec)
            preds.append(step_pred)
            dec_input = step_pred.detach()
        return torch.stack(preds, dim=1)


def train_model(X_train, y_train, X_val, y_val, n_feat, n_steps, n_targets,
                hidden_size, n_layers, dropout, device, label):
    model = MultiOutputLSTM(n_feat, n_steps, n_targets, hidden_size, n_layers, dropout).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BASE_CONFIG["batch_size"], shuffle=True)
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    optimizer = Adam(model.parameters(), lr=BASE_CONFIG["lr"],
                     weight_decay=BASE_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse_loss = nn.MSELoss()

    best_val_loss, best_state, patience_ctr = float("inf"), None, 0
    for epoch in range(BASE_CONFIG["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = mse_loss(model(xb), yb)
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), BASE_CONFIG["grad_clip"])
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

        if epoch % 10 == 0 or patience_ctr >= BASE_CONFIG["patience"]:
            print(f"    [{label}] epoch {epoch:3d}: val_loss={val_loss:.6f} "
                  f"best={best_val_loss:.6f} patience={patience_ctr}", flush=True)
        if patience_ctr >= BASE_CONFIG["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def run_config(hidden_size, n_layers, dropout, device, X_train, y_train,
               X_es, y_es, X_eval, y_eval, seq_eval, n_feat, n_steps, n_targets,
               target_cols, true_nano):
    label = f"h{hidden_size}_L{n_layers}"
    print(f"\n  === Config: {label} ===", flush=True)

    seed = BASE_CONFIG["seed"]
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    model = train_model(X_train, y_train, X_es, y_es, n_feat, n_steps, n_targets,
                        hidden_size, n_layers, dropout, device, label)

    with torch.no_grad():
        preds = []
        for start in range(0, len(X_eval), 512):
            xb = torch.from_numpy(X_eval[start:start + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred_mean = np.concatenate(preds, axis=0).mean(axis=1)
    y_eval_mean = y_eval.mean(axis=1)

    elapsed = time.time() - t0
    results = {}
    for i, name in enumerate(target_cols):
        mse = float(np.mean((pred_mean[:, i] - y_eval_mean[:, i]) ** 2))
        mae = float(np.mean(np.abs(pred_mean[:, i] - y_eval_mean[:, i])))
        corr = float(np.corrcoef(pred_mean[:, i], y_eval_mean[:, i])[0, 1])
        results[name] = {"mse": mse, "mae": mae, "r": corr}
        print(f"    {name:28s}: MSE={mse:.6f}  MAE={mae:.4f}  r={corr:.4f}", flush=True)

    print(f"    Time: {elapsed:.1f}s", flush=True)
    return results, pred_mean, y_eval_mean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
    feature_cols = BASE_CONFIG["features"]
    target_cols = BASE_CONFIG["targets"]
    n_targets = len(target_cols)

    H = BASE_CONFIG["horizon"]
    S = BASE_CONFIG["sub_window"]
    n_steps = H // S

    train_df, earlystop_df, within_val_df, cross_val_df = split_data(df)

    warmup = BASE_CONFIG["warmup_frames"]
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[warmup:])
    scaler = StandardScaler()
    scaler.fit(pd.concat(kept, ignore_index=True)[feature_cols].values)

    X_train, y_train, _ = build_windows(train_df, feature_cols, target_cols,
                                         BASE_CONFIG["train_stride"], scaler)
    X_es, y_es, _ = build_windows(earlystop_df, feature_cols, target_cols,
                                   BASE_CONFIG["eval_stride"], scaler)
    X_within, y_within, seq_within = build_windows(within_val_df, feature_cols, target_cols,
                                                    BASE_CONFIG["eval_stride"], scaler)
    X_cross, y_cross, seq_cross = build_windows(cross_val_df, feature_cols, target_cols,
                                                 BASE_CONFIG["eval_stride"], scaler)

    X_eval = np.concatenate([X_within, X_cross], axis=0)
    y_eval = np.concatenate([y_within, y_cross], axis=0)
    seq_eval = seq_within + seq_cross
    n_feat = len(feature_cols)

    # Get true_nano for CSV export (build windows with miss_rate_nano)
    nano_cols = ["miss_rate_nano"]
    _, y_nano_within, _ = build_windows(within_val_df, feature_cols, nano_cols,
                                         BASE_CONFIG["eval_stride"], scaler)
    _, y_nano_cross, _ = build_windows(cross_val_df, feature_cols, nano_cols,
                                        BASE_CONFIG["eval_stride"], scaler)
    true_nano = np.concatenate([y_nano_within, y_nano_cross], axis=0).mean(axis=1).squeeze()

    print(f"  Windows: train={len(y_train)}, earlystop={len(y_es)}, eval={len(y_eval)}", flush=True)

    configs = [
        (128, 2, 0.4),
        (256, 4, 0.4),
    ]

    all_results = {}
    for hidden, layers, dropout in configs:
        label = f"h{hidden}_L{layers}"
        res, pred_mean, y_eval_mean = run_config(
            hidden, layers, dropout, device,
            X_train, y_train, X_es, y_es, X_eval, y_eval, seq_eval,
            n_feat, n_steps, n_targets, target_cols, true_nano)
        all_results[label] = res

    # ── Side-by-side comparison ──
    ref = {"recovery_nano_to_small": 0.6440, "recovery_small_to_medium": 0.3150}

    print(f"\n{'='*90}")
    print(f"  CORRELATION COMPARISON: 3-output clipped vs 2-output configs")
    print(f"{'='*90}")
    print(f"  {'Target':28s} | {'3out clipped':>12s} | {'2out h128/L2':>12s} | {'2out h256/L4':>12s}")
    print(f"  {'-'*84}")
    for t in target_cols:
        r_ref = ref[t]
        r_h128 = all_results["h128_L2"][t]["r"]
        r_h256 = all_results["h256_L4"][t]["r"]
        d128 = r_h128 - r_ref
        d256 = r_h256 - r_ref
        print(f"  {t:28s} | {r_ref:>12.4f} | {r_h128:>7.4f} ({d128:+.4f}) | {r_h256:>7.4f} ({d256:+.4f})")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
