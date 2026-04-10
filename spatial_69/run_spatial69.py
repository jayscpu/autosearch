#!/usr/bin/env python3
"""
SPATIAL_69 vs SPATIAL_65: 3-seed comparison.

SPATIAL_69 = SPATIAL_65 + [optical_flow_magnitude_mean, optical_flow_magnitude_std,
                           optical_flow_direction_entropy, ssim_vs_prev]

Two configs (A=SPATIAL_65, B=SPATIAL_69), 3 seeds each (6 runs total).
Within-camera and cross-camera correlations reported SEPARATELY.
"""

import os
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

SPATIAL_69 = SPATIAL_65 + [
    "optical_flow_magnitude_mean",
    "optical_flow_magnitude_std",
    "optical_flow_direction_entropy",
    "ssim_vs_prev",
]

CONFIGS = {
    "SPATIAL_65": SPATIAL_65,
    "SPATIAL_69": SPATIAL_69,
}

TARGETS = ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"]
N_TARGETS = len(TARGETS)

HPARAMS = {
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
    "hidden_size": 128,
    "n_layers": 2,
    "dropout": 0.4,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 300,
    "patience": 40,
    "grad_clip": 1.0,
}

SEEDS = [42, 43, 44]


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(FEATURES_CSV)
    # Verify the 4 new columns exist
    required = ["optical_flow_magnitude_mean", "optical_flow_magnitude_std",
                "optical_flow_direction_entropy", "ssim_vs_prev"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns in CSV: {missing}", flush=True)
        sys.exit(1)
    df["miss_rate_nano"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    df["miss_rate_small"] = df["fn_small"] / df["x_count"].clip(lower=1)
    df["miss_rate_medium"] = df["fn_medium"] / df["x_count"].clip(lower=1)
    df["recovery_nano_to_small"] = df["miss_rate_nano"] - df["miss_rate_small"]
    df["recovery_small_to_medium"] = (df["miss_rate_small"] - df["miss_rate_medium"]).clip(lower=0)
    return df


def split_intersection(df, int_name, train_frac, es_frac):
    sub = df[df["sequence"] == int_name].sort_values("frame_id")
    n = len(sub)
    t_end = int(n * train_frac)
    e_end = int(n * es_frac)
    return sub.iloc[:t_end], sub.iloc[t_end:e_end], sub.iloc[e_end:]


def build_windows(df, feature_cols, target_cols, window, horizon, sub_window,
                  stride, scaler, warmup):
    n_steps = horizon // sub_window
    total = window + horizon
    X_list, y_list, seq_labels = [], [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        if len(sdf) < total:
            continue
        seq_name = sdf["sequence"].iloc[0]
        fv = scaler.transform(sdf[feature_cols].values)
        tv = sdf[target_cols].values
        for t in range(0, len(sdf) - total + 1, stride):
            X_list.append(fv[t:t + window])
            tg = np.zeros((n_steps, len(target_cols)), dtype=np.float32)
            for step in range(n_steps):
                s = t + window + step * sub_window
                tg[step] = tv[s:s + sub_window].mean(axis=0)
            y_list.append(tg)
            seq_labels.append(seq_name)
    if not X_list:
        return np.zeros((0, window, len(feature_cols)), dtype=np.float32), \
               np.zeros((0, horizon // sub_window, len(target_cols)), dtype=np.float32), []
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), seq_labels


def fit_scaler(train_df, feature_cols, warmup):
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[warmup:])
    scaler = StandardScaler()
    scaler.fit(pd.concat(kept, ignore_index=True)[feature_cols].values)
    return scaler


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, n_steps, n_targets, hidden, n_layers, dropout):
        super().__init__()
        self.n_steps, self.n_targets = n_steps, n_targets
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.decoder = nn.LSTMCell(input_size=n_targets, hidden_size=hidden)
        self.step_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
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
    model = MultiOutputLSTM(n_feat, n_steps, N_TARGETS,
                            HPARAMS["hidden_size"], HPARAMS["n_layers"],
                            HPARAMS["dropout"]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=HPARAMS["batch_size"], shuffle=True)
    vx, vy = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    opt = Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse = nn.MSELoss()
    best, best_state, best_epoch, pat = float("inf"), None, 0, 0
    for epoch in range(HPARAMS["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = mse(model(xb), yb)
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["grad_clip"])
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = mse(model(vx), vy).item()
        sched.step(vloss)
        if vloss < best:
            best = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            pat = 0
        else:
            pat += 1
        if epoch % 10 == 0 or pat >= HPARAMS["patience"]:
            print(f"    [{label}] ep{epoch}: val={vloss:.6f} best={best:.6f} pat={pat}",
                  flush=True)
        if pat >= HPARAMS["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_epoch


def predict(model, X, device):
    with torch.no_grad():
        preds = []
        for s in range(0, len(X), 512):
            xb = torch.from_numpy(X[s:s + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0).mean(axis=1)


def compute_correlations(pred, true, target_cols):
    metrics = {}
    for i, name in enumerate(target_cols):
        r = float(np.corrcoef(pred[:, i], true[:, i])[0, 1])
        metrics[name] = r
    return metrics


# ═══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════

def run_experiment(df, feat_name, feat_cols, device):
    W = HPARAMS["window"]
    H = HPARAMS["horizon"]
    SW = HPARAMS["sub_window"]
    n_steps = H // SW
    warmup = HPARAMS["warmup_frames"]
    n_feat = len(feat_cols)

    # Split data
    train_dfs, es_dfs, within_dfs = [], [], []
    for name in HPARAMS["train_intersections"]:
        tr, es, wv = split_intersection(df, name, HPARAMS["train_fraction"],
                                         HPARAMS["earlystop_fraction"])
        train_dfs.append(tr)
        es_dfs.append(es)
        within_dfs.append(wv)
    train_df = pd.concat(train_dfs, ignore_index=True)
    es_df = pd.concat(es_dfs, ignore_index=True)
    within_df = pd.concat(within_dfs, ignore_index=True)
    cross_df = df[df["sequence"] == HPARAMS["test_intersection"]].copy()

    scaler = fit_scaler(train_df, feat_cols, warmup)

    X_train, y_train_raw, _ = build_windows(train_df, feat_cols, TARGETS,
                                             W, H, SW, HPARAMS["train_stride"],
                                             scaler, warmup)
    X_es, y_es_raw, _ = build_windows(es_df, feat_cols, TARGETS,
                                       W, H, SW, HPARAMS["eval_stride"],
                                       scaler, warmup)
    X_within, y_within_raw, _ = build_windows(within_df, feat_cols, TARGETS,
                                               W, H, SW, HPARAMS["eval_stride"],
                                               scaler, warmup)
    X_cross, y_cross_raw, _ = build_windows(cross_df, feat_cols, TARGETS,
                                             W, H, SW, HPARAMS["eval_stride"],
                                             scaler, warmup)

    y_within_mean = y_within_raw.mean(axis=1)
    y_cross_mean = y_cross_raw.mean(axis=1)

    # Target normalization from training data
    y_flat = y_train_raw.reshape(-1, N_TARGETS)
    t_mean = y_flat.mean(axis=0)
    t_std = y_flat.std(axis=0)
    t_std[t_std < 1e-8] = 1.0

    y_train_norm = (y_train_raw - t_mean) / t_std
    y_es_norm = (y_es_raw - t_mean) / t_std

    print(f"\n  {feat_name} ({n_feat} features)")
    print(f"  Windows: train={len(y_train_raw)}, es={len(y_es_raw)}, "
          f"within={len(y_within_raw)}, cross={len(y_cross_raw)}", flush=True)

    results = []
    for seed in SEEDS:
        label = f"{feat_name}_s{seed}"
        print(f"\n  ── {feat_name} seed={seed} ──", flush=True)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        model, best_ep = train_model(X_train, y_train_norm, X_es, y_es_norm,
                                      n_feat, n_steps, device, label)

        pred_within_norm = predict(model, X_within, device)
        pred_cross_norm = predict(model, X_cross, device)

        pred_within = pred_within_norm * t_std + t_mean
        pred_cross = pred_cross_norm * t_std + t_mean

        elapsed = time.time() - t0

        within_r = compute_correlations(pred_within, y_within_mean, TARGETS)
        cross_r = compute_correlations(pred_cross, y_cross_mean, TARGETS)

        print(f"    Within-camera (n={len(y_within_mean)}):", flush=True)
        for tgt in TARGETS:
            print(f"      {tgt:28s}: r={within_r[tgt]:.4f}", flush=True)
        print(f"    Cross-camera (n={len(y_cross_mean)}):", flush=True)
        for tgt in TARGETS:
            print(f"      {tgt:28s}: r={cross_r[tgt]:.4f}", flush=True)
        print(f"    Best epoch: {best_ep}, time: {elapsed:.1f}s", flush=True)

        results.append({
            "features": feat_name,
            "seed": seed,
            "within_r_nano": round(within_r["miss_rate_nano"], 6),
            "within_r_ns": round(within_r["recovery_nano_to_small"], 6),
            "within_r_sm": round(within_r["recovery_small_to_medium"], 6),
            "cross_r_nano": round(cross_r["miss_rate_nano"], 6),
            "cross_r_ns": round(cross_r["recovery_nano_to_small"], 6),
            "cross_r_sm": round(cross_r["recovery_small_to_medium"], 6),
            "best_epoch": best_ep,
            "elapsed_s": round(elapsed, 1),
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = load_data()
    print(f"Data loaded: {len(df)} rows", flush=True)

    all_results = []
    for feat_name, feat_cols in CONFIGS.items():
        results = run_experiment(df, feat_name, feat_cols, device)
        all_results.extend(results)

    # Save results TSV
    results_df = pd.DataFrame(all_results)
    tsv_path = SCRIPT_DIR / "results.tsv"
    results_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSaved {tsv_path}", flush=True)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY: SPATIAL_65 vs SPATIAL_69 (mean ± std over 3 seeds)")
    print(f"{'='*80}")

    metric_cols = ["within_r_nano", "within_r_ns", "within_r_sm",
                   "cross_r_nano", "cross_r_ns", "cross_r_sm"]

    summary_rows = {}
    for feat_name in CONFIGS:
        sub = results_df[results_df["features"] == feat_name]
        row = {}
        for col in metric_cols:
            vals = sub[col].values
            row[f"{col}_mean"] = np.mean(vals)
            row[f"{col}_std"] = np.std(vals)
        summary_rows[feat_name] = row

    header = f"  {'metric':>20s}  {'SPATIAL_65':>18s}  {'SPATIAL_69':>18s}  {'delta':>10s}"
    print(header)
    print(f"  {'-'*72}")
    for col in metric_cols:
        m65 = summary_rows["SPATIAL_65"][f"{col}_mean"]
        s65 = summary_rows["SPATIAL_65"][f"{col}_std"]
        m69 = summary_rows["SPATIAL_69"][f"{col}_mean"]
        s69 = summary_rows["SPATIAL_69"][f"{col}_std"]
        delta = m69 - m65
        sign = "+" if delta >= 0 else ""
        print(f"  {col:>20s}  {m65:+.4f}±{s65:.4f}  {m69:+.4f}±{s69:.4f}  {sign}{delta:.4f}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
