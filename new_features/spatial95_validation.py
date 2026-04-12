#!/usr/bin/env python3
"""3-seed validation of SPATIAL_95 (65 original + 30 new spatial features).

Mirrors final_validation/run_experiments.py experiment1 exactly, but:
  - Features: SPATIAL_95
  - Data: new_features/pod_features_merged_95.csv
  - Outputs: new_features/spatial95_validation.tsv,
             new_features/predictions_95_seed{42,43,44}.csv
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
FEATURES_CSV = SCRIPT_DIR / "pod_features_merged_95.csv"

sys.path.insert(0, str(PROJECT_DIR / "second_autosearch" / "src"))
from pod_features import SPATIAL_65  # noqa: E402

NEW_FEATURES_30 = [
    "blob_small_fraction", "blob_large_fraction", "blob_size_cv",
    "blob_compactness_mean", "blob_compactness_std",
    "blob_aspect_ratio_mean", "blob_aspect_ratio_std",
    "blob_nn_dist_mean", "blob_nn_dist_std", "blob_overlap_count",
    "fg_spatial_entropy", "fg_center_of_mass_y", "fg_center_of_mass_x",
    "fg_quadrant_imbalance", "fg_perimeter_ratio",
    "motion_spatial_entropy", "motion_concentration", "motion_vertical_bias",
    "temporal_diff_skewness", "temporal_diff_kurtosis",
    "contour_count", "contour_small_fraction", "contour_large_fraction",
    "contour_closed_ratio", "contour_hierarchy_depth",
    "spectral_centroid", "spectral_rolloff_85",
    "band_energy_fine_over_coarse", "subband_entropy", "feature_congestion",
]
SPATIAL_95 = SPATIAL_65 + NEW_FEATURES_30
assert len(SPATIAL_95) == 95

CONFIG = {
    "features": SPATIAL_95,
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
N_TARGETS = len(CONFIG["targets"])


def load_data():
    df = pd.read_csv(FEATURES_CSV)
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


def build_windows(df, feat_cols, tgt_cols, window, horizon, sub_window,
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
        fv = scaler.transform(sdf[feat_cols].values)
        tv = sdf[tgt_cols].values
        for t in range(0, len(sdf) - total + 1, stride):
            X_list.append(fv[t:t + window])
            tg = np.zeros((n_steps, len(tgt_cols)), dtype=np.float32)
            for step in range(n_steps):
                s = t + window + step * sub_window
                tg[step] = tv[s:s + sub_window].mean(axis=0)
            y_list.append(tg)
            seq_labels.append(seq_name)
    if not X_list:
        return (np.zeros((0, window, len(feat_cols)), dtype=np.float32),
                np.zeros((0, n_steps, len(tgt_cols)), dtype=np.float32), [])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), seq_labels


def fit_scaler(train_df, feat_cols, warmup):
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[warmup:])
    scaler = StandardScaler()
    scaler.fit(pd.concat(kept, ignore_index=True)[feat_cols].values)
    return scaler


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
                            CONFIG["hidden_size"], CONFIG["n_layers"],
                            CONFIG["dropout"]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)
    vx, vy = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    opt = Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse = nn.MSELoss()
    best, best_state, best_epoch, pat = float("inf"), None, 0, 0
    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = mse(model(xb), yb)
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
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
        if epoch % 10 == 0 or pat >= CONFIG["patience"]:
            print(f"    [{label}] ep{epoch}: val={vloss:.6f} best={best:.6f} pat={pat}",
                  flush=True)
        if pat >= CONFIG["patience"]:
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


def compute_metrics(pred, true, tgt_cols):
    metrics = {}
    for i, name in enumerate(tgt_cols):
        r = float(np.corrcoef(pred[:, i], true[:, i])[0, 1])
        mse_val = float(np.mean((pred[:, i] - true[:, i]) ** 2))
        metrics[f"r_{name}"] = r
        metrics[f"mse_{name}"] = mse_val
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)
    print(f"  SPATIAL_95 length: {len(SPATIAL_95)}", flush=True)

    df = load_data()
    print(f"  Data loaded: {len(df)} rows", flush=True)

    feat_cols = CONFIG["features"]
    tgt_cols = CONFIG["targets"]
    W, H, SW = CONFIG["window"], CONFIG["horizon"], CONFIG["sub_window"]
    n_steps = H // SW
    warmup = CONFIG["warmup_frames"]

    train_dfs, es_dfs, within_dfs = [], [], []
    for name in CONFIG["train_intersections"]:
        tr, es, wv = split_intersection(df, name, CONFIG["train_fraction"],
                                         CONFIG["earlystop_fraction"])
        train_dfs.append(tr); es_dfs.append(es); within_dfs.append(wv)
    train_df = pd.concat(train_dfs, ignore_index=True)
    es_df = pd.concat(es_dfs, ignore_index=True)
    within_df = pd.concat(within_dfs, ignore_index=True)
    cross_df = df[df["sequence"] == CONFIG["test_intersection"]].copy()

    scaler = fit_scaler(train_df, feat_cols, warmup)
    X_train, y_train_raw, _ = build_windows(train_df, feat_cols, tgt_cols,
                                             W, H, SW, CONFIG["train_stride"],
                                             scaler, warmup)
    X_es, y_es_raw, _ = build_windows(es_df, feat_cols, tgt_cols,
                                       W, H, SW, CONFIG["eval_stride"], scaler, warmup)
    X_within, y_within_raw, seq_within = build_windows(
        within_df, feat_cols, tgt_cols, W, H, SW, CONFIG["eval_stride"], scaler, warmup)
    X_cross, y_cross_raw, seq_cross = build_windows(
        cross_df, feat_cols, tgt_cols, W, H, SW, CONFIG["eval_stride"], scaler, warmup)

    y_within_mean = y_within_raw.mean(axis=1)
    y_cross_mean = y_cross_raw.mean(axis=1)

    y_flat = y_train_raw.reshape(-1, N_TARGETS)
    t_mean = y_flat.mean(axis=0)
    t_std = y_flat.std(axis=0); t_std[t_std < 1e-8] = 1.0

    print(f"  Target norm: mean={t_mean}  std={t_std}", flush=True)
    y_train_norm = (y_train_raw - t_mean) / t_std
    y_es_norm = (y_es_raw - t_mean) / t_std

    n_feat = len(feat_cols)
    print(f"  Windows: train={len(y_train_raw)}, es={len(y_es_raw)}, "
          f"within={len(y_within_raw)}, cross={len(y_cross_raw)}", flush=True)

    seeds = [42, 43, 44]
    all_results = []
    for seed in seeds:
        label = f"seed{seed}"
        print(f"\n  ── Seed {seed} ──", flush=True)
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        model, best_ep = train_model(X_train, y_train_norm, X_es, y_es_norm,
                                      n_feat, n_steps, device, label)
        pred_within = predict(model, X_within, device) * t_std + t_mean
        pred_cross = predict(model, X_cross, device) * t_std + t_mean
        elapsed = time.time() - t0

        wm = compute_metrics(pred_within, y_within_mean, tgt_cols)
        cm = compute_metrics(pred_cross, y_cross_mean, tgt_cols)

        for split, m, n in [("within", wm, len(y_within_mean)),
                             ("cross", cm, len(y_cross_mean))]:
            print(f"    {split} (n={n}):", flush=True)
            for name in tgt_cols:
                print(f"      {name:28s}: r={m[f'r_{name}']:.4f}  MSE={m[f'mse_{name}']:.6f}",
                      flush=True)
        print(f"    best_ep={best_ep}  time={elapsed:.1f}s", flush=True)

        rows = []
        for i in range(len(y_within_mean)):
            rows.append({"frame_idx": i, "split": "within",
                         "pred_nano": round(float(pred_within[i, 0]), 6),
                         "pred_recovery_ns": round(float(pred_within[i, 1]), 6),
                         "pred_recovery_sm": round(float(pred_within[i, 2]), 6),
                         "true_nano": round(float(y_within_mean[i, 0]), 6),
                         "true_recovery_ns": round(float(y_within_mean[i, 1]), 6),
                         "true_recovery_sm": round(float(y_within_mean[i, 2]), 6),
                         "intersection": seq_within[i]})
        for i in range(len(y_cross_mean)):
            rows.append({"frame_idx": i, "split": "cross",
                         "pred_nano": round(float(pred_cross[i, 0]), 6),
                         "pred_recovery_ns": round(float(pred_cross[i, 1]), 6),
                         "pred_recovery_sm": round(float(pred_cross[i, 2]), 6),
                         "true_nano": round(float(y_cross_mean[i, 0]), 6),
                         "true_recovery_ns": round(float(y_cross_mean[i, 1]), 6),
                         "true_recovery_sm": round(float(y_cross_mean[i, 2]), 6),
                         "intersection": seq_cross[i] if i < len(seq_cross) else CONFIG["test_intersection"]})
        csv_path = SCRIPT_DIR / f"predictions_95_seed{seed}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"    Saved {csv_path}", flush=True)

        result = {"seed": seed, "best_epoch": best_ep, "elapsed_s": round(elapsed, 1)}
        for k, v in wm.items(): result[f"within_{k}"] = round(v, 6)
        for k, v in cm.items(): result[f"cross_{k}"] = round(v, 6)
        all_results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n  3-SEED SUMMARY (mean ± std)", flush=True)
    summary = {"config": "spatial95_clip_sm_normed_h128_L2"}
    for split in ["within", "cross"]:
        print(f"  {split}:", flush=True)
        for name in tgt_cols:
            r_vals = [r[f"{split}_r_{name}"] for r in all_results]
            mse_vals = [r[f"{split}_mse_{name}"] for r in all_results]
            rm, rs = np.mean(r_vals), np.std(r_vals)
            mm, ms = np.mean(mse_vals), np.std(mse_vals)
            print(f"    {name:28s}: r={rm:.4f}±{rs:.4f}  MSE={mm:.6f}±{ms:.6f}", flush=True)
            summary[f"{split}_r_{name}_mean"] = round(rm, 6)
            summary[f"{split}_r_{name}_std"] = round(rs, 6)
            summary[f"{split}_mse_{name}_mean"] = round(mm, 6)
            summary[f"{split}_mse_{name}_std"] = round(ms, 6)

    tsv_path = SCRIPT_DIR / "spatial95_validation.tsv"
    pd.DataFrame(all_results).to_csv(tsv_path, sep="\t", index=False)
    with open(tsv_path, "a") as f:
        f.write(f"\n# Summary (mean±std): {summary}\n")
    print(f"  Saved {tsv_path}", flush=True)

    # Comparison table
    baseline = {
        "within_r_miss_rate_nano": (0.848, 0.003),
        "within_r_recovery_nano_to_small": (0.717, 0.024),
        "within_r_recovery_small_to_medium": (0.255, 0.044),
        "cross_r_miss_rate_nano": (0.449, 0.024),
        "cross_r_recovery_nano_to_small": (0.496, 0.048),
        "cross_r_recovery_small_to_medium": (0.330, 0.035),
    }
    header = "Config      | within_r_nano | within_r_ns | within_r_sm | cross_r_nano | cross_r_ns | cross_r_sm"
    print("\n" + header, flush=True)
    sm_b = lambda k: f"{baseline[k][0]:.3f}±{baseline[k][1]:.3f}"
    print(f"SPATIAL_65  | {sm_b('within_r_miss_rate_nano'):13s} | "
          f"{sm_b('within_r_recovery_nano_to_small'):11s} | "
          f"{sm_b('within_r_recovery_small_to_medium'):11s} | "
          f"{sm_b('cross_r_miss_rate_nano'):12s} | "
          f"{sm_b('cross_r_recovery_nano_to_small'):10s} | "
          f"{sm_b('cross_r_recovery_small_to_medium'):10s}", flush=True)

    s95 = {}
    for split, short in [("within", ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"])]:
        pass

    def pair(split, name):
        m = summary[f"{split}_r_{name}_mean"]; s = summary[f"{split}_r_{name}_std"]
        return m, s

    parts = []
    for split, name in [("within", "miss_rate_nano"), ("within", "recovery_nano_to_small"),
                         ("within", "recovery_small_to_medium"),
                         ("cross", "miss_rate_nano"), ("cross", "recovery_nano_to_small"),
                         ("cross", "recovery_small_to_medium")]:
        m, s = pair(split, name)
        parts.append((m, s, f"{split}_r_{name}"))
    row95 = "SPATIAL_95  | " + " | ".join(
        f"{m:.3f}±{s:.3f}".ljust(13 if i == 0 else 11 if i < 3 else 12 if i == 3 else 10)
        for i, (m, s, _) in enumerate(parts))
    print(row95, flush=True)

    key_map = [
        ("within_r_miss_rate_nano", ("within", "miss_rate_nano")),
        ("within_r_recovery_nano_to_small", ("within", "recovery_nano_to_small")),
        ("within_r_recovery_small_to_medium", ("within", "recovery_small_to_medium")),
        ("cross_r_miss_rate_nano", ("cross", "miss_rate_nano")),
        ("cross_r_recovery_nano_to_small", ("cross", "recovery_nano_to_small")),
        ("cross_r_recovery_small_to_medium", ("cross", "recovery_small_to_medium")),
    ]
    deltas = []
    flags = []
    for bk, (split, name) in key_map:
        bm = baseline[bk][0]; nm = summary[f"{split}_r_{name}_mean"]
        d = nm - bm
        deltas.append(d)
        if d > 0.02: flags.append(f"  ↑ IMPROVED {bk}: +{d:.3f}")
        elif d < -0.02: flags.append(f"  ↓ DEGRADED {bk}: {d:.3f}")
    dstr = "Delta       | " + " | ".join(
        f"{'+' if d >= 0 else ''}{d:.3f}".ljust(13 if i == 0 else 11 if i < 3 else 12 if i == 3 else 10)
        for i, d in enumerate(deltas))
    print(dstr, flush=True)

    if flags:
        print("\nSignificant changes (>|0.02|):", flush=True)
        for f in flags: print(f, flush=True)
    else:
        print("\nNo changes >|0.02|.", flush=True)


if __name__ == "__main__":
    main()
