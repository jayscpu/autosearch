#!/usr/bin/env python3
"""
3-seed validation for clip_sm and clip_sm_normed configs.
Runs each config with seeds 42, 43, 44, saves per-seed CSVs,
prints per-seed correlations, and writes summary TSV.
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "second_autosearch" / "data"
FEATURES_CSV = DATA_DIR / "pod_features_all.csv"

sys.path.insert(0, str(PROJECT_DIR / "second_autosearch" / "src"))
from pod_features import SPATIAL_65

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
}

N_TARGETS = len(CONFIG["targets"])
SEEDS = [42, 43, 44]


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
    for int_name in CONFIG["train_intersections"]:
        sub = df[df["sequence"] == int_name].sort_values("frame_id")
        n = len(sub)
        train_end = int(n * CONFIG["train_fraction"])
        earlystop_end = int(n * CONFIG["earlystop_fraction"])
        train_dfs.append(sub.iloc[:train_end])
        earlystop_dfs.append(sub.iloc[train_end:earlystop_end])
        within_val_dfs.append(sub.iloc[earlystop_end:])
    return (pd.concat(train_dfs, ignore_index=True),
            pd.concat(earlystop_dfs, ignore_index=True),
            pd.concat(within_val_dfs, ignore_index=True),
            df[df["sequence"] == CONFIG["test_intersection"]].copy())


def build_windows(df, feature_cols, target_cols, stride, scaler):
    W, H, S = CONFIG["window"], CONFIG["horizon"], CONFIG["sub_window"]
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
        target_vals = sdf[target_cols].values
        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t:t + W])
            targets = np.zeros((n_steps, len(target_cols)), dtype=np.float32)
            for step in range(n_steps):
                start = t + W + step * S
                targets[step] = target_vals[start:start + S].mean(axis=0)
            y_list.append(targets)
            seq_labels.append(seq_name)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), seq_labels


class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, n_steps, n_targets):
        super().__init__()
        hidden = CONFIG["hidden_size"]
        n_layers = CONFIG["n_layers"]
        dropout = CONFIG["dropout"] if n_layers > 1 else 0.0
        self.n_steps, self.n_targets = n_steps, n_targets
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTMCell(input_size=n_targets, hidden_size=hidden)
        self.step_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
            nn.Linear(64, n_targets))

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


def train_model(X_train, y_train, X_val, y_val, n_feat, n_steps, device, label):
    model = MultiOutputLSTM(n_feat, n_steps, N_TARGETS).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)
    val_x, val_y = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    optimizer = Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
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
            print(f"    [{label}] epoch {epoch:3d}: val={val_loss:.6f} "
                  f"best={best_val_loss:.6f} pat={patience_ctr}", flush=True)
        if patience_ctr >= CONFIG["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def run_single(seed, normalize, X_train, y_train_raw, X_es, y_es_raw,
               X_eval, y_eval_mean, seq_eval, n_feat, n_steps, device,
               target_cols, out_path, label):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if normalize:
        y_flat = y_train_raw.reshape(-1, N_TARGETS)
        t_mean = y_flat.mean(axis=0)
        t_std = y_flat.std(axis=0)
        t_std[t_std < 1e-8] = 1.0
        y_train = (y_train_raw - t_mean) / t_std
        y_es = (y_es_raw - t_mean) / t_std
    else:
        y_train, y_es = y_train_raw, y_es_raw
        t_mean, t_std = None, None

    t0 = time.time()
    model = train_model(y_train=y_train, X_train=X_train, X_val=X_es, y_val=y_es,
                        n_feat=n_feat, n_steps=n_steps, device=device, label=label)

    with torch.no_grad():
        preds = []
        for start in range(0, len(X_eval), 512):
            xb = torch.from_numpy(X_eval[start:start + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred_raw = np.concatenate(preds, axis=0).mean(axis=1)

    if normalize:
        pred_mean = pred_raw * t_std + t_mean
    else:
        pred_mean = pred_raw

    elapsed = time.time() - t0
    corrs = {}
    for i, name in enumerate(target_cols):
        mse = float(np.mean((pred_mean[:, i] - y_eval_mean[:, i]) ** 2))
        corr = float(np.corrcoef(pred_mean[:, i], y_eval_mean[:, i])[0, 1])
        corrs[name] = corr
        print(f"    {name:28s}: MSE={mse:.6f}  r={corr:.4f}", flush=True)
    print(f"    Time: {elapsed:.1f}s", flush=True)

    # Save CSV
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
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    Saved {out_path}", flush=True)
    return corrs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
    feature_cols = CONFIG["features"]
    target_cols = CONFIG["targets"]
    n_steps = CONFIG["horizon"] // CONFIG["sub_window"]

    train_df, earlystop_df, within_val_df, cross_val_df = split_data(df)

    warmup = CONFIG["warmup_frames"]
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[warmup:])
    scaler = StandardScaler()
    scaler.fit(pd.concat(kept, ignore_index=True)[feature_cols].values)

    X_train, y_train, _ = build_windows(train_df, feature_cols, target_cols,
                                         CONFIG["train_stride"], scaler)
    X_es, y_es, _ = build_windows(earlystop_df, feature_cols, target_cols,
                                   CONFIG["eval_stride"], scaler)
    X_within, y_within, seq_within = build_windows(within_val_df, feature_cols, target_cols,
                                                    CONFIG["eval_stride"], scaler)
    X_cross, y_cross, seq_cross = build_windows(cross_val_df, feature_cols, target_cols,
                                                 CONFIG["eval_stride"], scaler)

    X_eval = np.concatenate([X_within, X_cross], axis=0)
    y_eval = np.concatenate([y_within, y_cross], axis=0)
    seq_eval = seq_within + seq_cross
    y_eval_mean = y_eval.mean(axis=1)
    n_feat = len(feature_cols)

    print(f"  Windows: train={len(y_train)}, es={len(y_es)}, eval={len(y_eval)}", flush=True)

    configs = [
        ("clip_sm", False),
        ("clip_sm_normed", True),
    ]

    all_corrs = {}  # config_name -> {seed -> {target -> r}}

    for config_name, normalize in configs:
        all_corrs[config_name] = {}
        for seed in SEEDS:
            label = f"{config_name}_s{seed}"
            out_path = SCRIPT_DIR / f"predictions_{config_name}_seed{seed}.csv"
            print(f"\n  === {label} ===", flush=True)
            corrs = run_single(seed, normalize, X_train, y_train, X_es, y_es,
                               X_eval, y_eval_mean, seq_eval, n_feat, n_steps,
                               device, target_cols, out_path, label)
            all_corrs[config_name][seed] = corrs

    # ── Per-seed table ──
    print(f"\n{'='*100}")
    print(f"  PER-SEED CORRELATIONS")
    print(f"{'='*100}")
    print(f"  {'Config':20s} {'Seed':>5s} | {'nano (r)':>10s} | {'rec_ns (r)':>10s} | {'rec_sm (r)':>10s}")
    print(f"  {'-'*90}")
    for config_name in ["clip_sm", "clip_sm_normed"]:
        for seed in SEEDS:
            c = all_corrs[config_name][seed]
            print(f"  {config_name:20s} {seed:>5d} | "
                  f"{c['miss_rate_nano']:>10.4f} | "
                  f"{c['recovery_nano_to_small']:>10.4f} | "
                  f"{c['recovery_small_to_medium']:>10.4f}")
        print(f"  {'-'*90}")

    # ── Mean ± std summary ──
    print(f"\n{'='*100}")
    print(f"  MEAN ± STD (3 seeds)")
    print(f"{'='*100}")
    print(f"  {'Config':20s} | {'nano (r)':>16s} | {'rec_ns (r)':>16s} | {'rec_sm (r)':>16s}")
    print(f"  {'-'*76}")

    summary_rows = []
    for config_name in ["clip_sm", "clip_sm_normed"]:
        means, stds = {}, {}
        for target in target_cols:
            vals = [all_corrs[config_name][s][target] for s in SEEDS]
            means[target] = np.mean(vals)
            stds[target] = np.std(vals)
        print(f"  {config_name:20s} | "
              f"{means['miss_rate_nano']:.4f} ± {stds['miss_rate_nano']:.4f} | "
              f"{means['recovery_nano_to_small']:.4f} ± {stds['recovery_nano_to_small']:.4f} | "
              f"{means['recovery_small_to_medium']:.4f} ± {stds['recovery_small_to_medium']:.4f}")
        summary_rows.append({
            "config": config_name,
            "nano_r_mean": round(means["miss_rate_nano"], 4),
            "nano_r_std": round(stds["miss_rate_nano"], 4),
            "rec_ns_r_mean": round(means["recovery_nano_to_small"], 4),
            "rec_ns_r_std": round(stds["recovery_nano_to_small"], 4),
            "rec_sm_r_mean": round(means["recovery_small_to_medium"], 4),
            "rec_sm_r_std": round(stds["recovery_small_to_medium"], 4),
        })
    print(f"{'='*100}")

    # ── Save TSV ──
    tsv_path = SCRIPT_DIR / "validation_3seed.tsv"
    pd.DataFrame(summary_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Summary saved: {tsv_path}", flush=True)


if __name__ == "__main__":
    main()
