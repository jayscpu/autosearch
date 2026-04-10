#!/usr/bin/env python3
"""
Final validation experiments for clip_sm_normed config (h128/L2).

Experiment 1: 3-seed validation (seeds 42, 43, 44)
  - Within-camera and cross-camera metrics reported SEPARATELY
  - Per-seed predictions saved to CSV
  - Summary saved to final_3seed_validation.tsv

Experiment 2: Leave-one-camera-out 4-fold cross-validation (seed 42)
  - Each fold holds out one training intersection
  - Evaluates on held-out camera AND SE38th
  - Results saved to kfold_leave_one_camera_out.tsv
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


def _fsync_write(path, mode, content):
    with open(path, mode) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════

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
    return np.concatenate(preds, axis=0).mean(axis=1)  # (N, N_TARGETS)


def compute_metrics(pred, true, target_cols):
    """Return dict of r and mse per target."""
    metrics = {}
    for i, name in enumerate(target_cols):
        r = float(np.corrcoef(pred[:, i], true[:, i])[0, 1])
        mse_val = float(np.mean((pred[:, i] - true[:, i]) ** 2))
        metrics[f"r_{name}"] = r
        metrics[f"mse_{name}"] = mse_val
    return metrics


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: 3-seed validation
# ═══════════════════════════════════════════════════════════════════

def experiment1(df, device):
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 1: 3-seed validation (clip_sm_normed, h128/L2)")
    print(f"{'='*80}\n")

    feat_cols = CONFIG["features"]
    tgt_cols = CONFIG["targets"]
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    SW = CONFIG["sub_window"]
    n_steps = H // SW
    warmup = CONFIG["warmup_frames"]

    # Split data
    train_dfs, es_dfs, within_dfs = [], [], []
    for name in CONFIG["train_intersections"]:
        tr, es, wv = split_intersection(df, name, CONFIG["train_fraction"],
                                         CONFIG["earlystop_fraction"])
        train_dfs.append(tr)
        es_dfs.append(es)
        within_dfs.append(wv)
    train_df = pd.concat(train_dfs, ignore_index=True)
    es_df = pd.concat(es_dfs, ignore_index=True)
    within_df = pd.concat(within_dfs, ignore_index=True)
    cross_df = df[df["sequence"] == CONFIG["test_intersection"]].copy()

    scaler = fit_scaler(train_df, feat_cols, warmup)

    X_train, y_train_raw, _ = build_windows(train_df, feat_cols, tgt_cols,
                                             W, H, SW, CONFIG["train_stride"],
                                             scaler, warmup)
    X_es, y_es_raw, _ = build_windows(es_df, feat_cols, tgt_cols,
                                       W, H, SW, CONFIG["eval_stride"],
                                       scaler, warmup)
    X_within, y_within_raw, seq_within = build_windows(within_df, feat_cols, tgt_cols,
                                                        W, H, SW, CONFIG["eval_stride"],
                                                        scaler, warmup)
    X_cross, y_cross_raw, seq_cross = build_windows(cross_df, feat_cols, tgt_cols,
                                                     W, H, SW, CONFIG["eval_stride"],
                                                     scaler, warmup)

    y_within_mean = y_within_raw.mean(axis=1)
    y_cross_mean = y_cross_raw.mean(axis=1)

    # Target normalization from training data
    y_flat = y_train_raw.reshape(-1, N_TARGETS)
    t_mean = y_flat.mean(axis=0)
    t_std = y_flat.std(axis=0)
    t_std[t_std < 1e-8] = 1.0

    print(f"  Target normalization (from training):", flush=True)
    for i, name in enumerate(tgt_cols):
        print(f"    {name:28s}: mean={t_mean[i]:.6f}  std={t_std[i]:.6f}", flush=True)

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
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        model, best_ep = train_model(X_train, y_train_norm, X_es, y_es_norm,
                                      n_feat, n_steps, device, label)

        # Predict on within and cross SEPARATELY
        pred_within_norm = predict(model, X_within, device)
        pred_cross_norm = predict(model, X_cross, device)

        # Denormalize
        pred_within = pred_within_norm * t_std + t_mean
        pred_cross = pred_cross_norm * t_std + t_mean

        elapsed = time.time() - t0

        # Compute metrics separately
        within_metrics = compute_metrics(pred_within, y_within_mean, tgt_cols)
        cross_metrics = compute_metrics(pred_cross, y_cross_mean, tgt_cols)

        print(f"\n    Within-camera (n={len(y_within_mean)}):", flush=True)
        for name in tgt_cols:
            print(f"      {name:28s}: r={within_metrics[f'r_{name}']:.4f}  "
                  f"MSE={within_metrics[f'mse_{name}']:.6f}", flush=True)

        print(f"    Cross-camera (n={len(y_cross_mean)}):", flush=True)
        for name in tgt_cols:
            print(f"      {name:28s}: r={cross_metrics[f'r_{name}']:.4f}  "
                  f"MSE={cross_metrics[f'mse_{name}']:.6f}", flush=True)

        print(f"    Best epoch: {best_ep}, time: {elapsed:.1f}s", flush=True)

        # Save per-seed CSV with within and cross predictions
        rows = []
        for i in range(len(y_within_mean)):
            rows.append({
                "frame_idx": i, "split": "within",
                "pred_nano": round(float(pred_within[i, 0]), 6),
                "pred_recovery_ns": round(float(pred_within[i, 1]), 6),
                "pred_recovery_sm": round(float(pred_within[i, 2]), 6),
                "true_nano": round(float(y_within_mean[i, 0]), 6),
                "true_recovery_ns": round(float(y_within_mean[i, 1]), 6),
                "true_recovery_sm": round(float(y_within_mean[i, 2]), 6),
                "intersection": seq_within[i],
            })
        for i in range(len(y_cross_mean)):
            rows.append({
                "frame_idx": i, "split": "cross",
                "pred_nano": round(float(pred_cross[i, 0]), 6),
                "pred_recovery_ns": round(float(pred_cross[i, 1]), 6),
                "pred_recovery_sm": round(float(pred_cross[i, 2]), 6),
                "true_nano": round(float(y_cross_mean[i, 0]), 6),
                "true_recovery_ns": round(float(y_cross_mean[i, 1]), 6),
                "true_recovery_sm": round(float(y_cross_mean[i, 2]), 6),
                "intersection": seq_cross[i] if i < len(seq_cross) else CONFIG["test_intersection"],
            })
        csv_path = SCRIPT_DIR / f"predictions_final_seed{seed}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"    Saved {csv_path}", flush=True)

        result = {"seed": seed, "best_epoch": best_ep, "elapsed_s": round(elapsed, 1)}
        for k, v in within_metrics.items():
            result[f"within_{k}"] = round(v, 6)
        for k, v in cross_metrics.items():
            result[f"cross_{k}"] = round(v, 6)
        all_results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary: mean ± std
    print(f"\n{'='*80}")
    print(f"  3-SEED SUMMARY (mean ± std)")
    print(f"{'='*80}")

    summary = {"config": "clip_sm_normed_h128_L2"}
    for split in ["within", "cross"]:
        print(f"\n  {split.upper()}-camera:", flush=True)
        for name in tgt_cols:
            r_key = f"{split}_r_{name}"
            mse_key = f"{split}_mse_{name}"
            r_vals = [r[r_key] for r in all_results]
            mse_vals = [r[mse_key] for r in all_results]
            r_mean, r_std = np.mean(r_vals), np.std(r_vals)
            mse_mean, mse_std = np.mean(mse_vals), np.std(mse_vals)
            print(f"    {name:28s}: r={r_mean:.4f}±{r_std:.4f}  "
                  f"MSE={mse_mean:.6f}±{mse_std:.6f}", flush=True)
            summary[f"{split}_r_{name}_mean"] = round(r_mean, 6)
            summary[f"{split}_r_{name}_std"] = round(r_std, 6)
            summary[f"{split}_mse_{name}_mean"] = round(mse_mean, 6)
            summary[f"{split}_mse_{name}_std"] = round(mse_std, 6)

    # Save TSV
    tsv_path = SCRIPT_DIR / "final_3seed_validation.tsv"
    # Per-seed rows + summary row
    per_seed_df = pd.DataFrame(all_results)
    per_seed_df.to_csv(tsv_path, sep="\t", index=False)
    # Append summary
    with open(tsv_path, "a") as f:
        f.write(f"\n# Summary (mean±std): {summary}\n")
    print(f"\n  Saved {tsv_path}", flush=True)

    return all_results


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Leave-one-camera-out 4-fold CV
# ═══════════════════════════════════════════════════════════════════

def experiment2(df, device):
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 2: Leave-one-camera-out 4-fold CV (seed=42)")
    print(f"{'='*80}\n")

    feat_cols = CONFIG["features"]
    tgt_cols = CONFIG["targets"]
    W = CONFIG["window"]
    H = CONFIG["horizon"]
    SW = CONFIG["sub_window"]
    n_steps = H // SW
    n_feat = len(feat_cols)
    warmup = CONFIG["warmup_frames"]
    seed = 42

    all_intersections = CONFIG["train_intersections"]
    cross_df_full = df[df["sequence"] == CONFIG["test_intersection"]].copy()

    fold_results = []

    for fold_idx, held_out in enumerate(all_intersections):
        train_cams = [c for c in all_intersections if c != held_out]
        fold_label = f"fold{fold_idx+1}_{held_out.split('_')[-1]}"
        print(f"\n  ── Fold {fold_idx+1}: hold out {held_out} ──", flush=True)
        print(f"    Train on: {[c.split('_')[-1] for c in train_cams]}", flush=True)

        # Split each training camera temporally
        train_dfs, es_dfs = [], []
        for cam in train_cams:
            tr, es, _ = split_intersection(df, cam, CONFIG["train_fraction"],
                                            CONFIG["earlystop_fraction"])
            train_dfs.append(tr)
            es_dfs.append(es)
        train_df = pd.concat(train_dfs, ignore_index=True)
        es_df = pd.concat(es_dfs, ignore_index=True)

        # Held-out camera: use ALL of it for evaluation
        held_out_df = df[df["sequence"] == held_out].copy()

        scaler = fit_scaler(train_df, feat_cols, warmup)

        X_train, y_train_raw, _ = build_windows(train_df, feat_cols, tgt_cols,
                                                  W, H, SW, CONFIG["train_stride"],
                                                  scaler, warmup)
        X_es, y_es_raw, _ = build_windows(es_df, feat_cols, tgt_cols,
                                           W, H, SW, CONFIG["eval_stride"],
                                           scaler, warmup)
        X_held, y_held_raw, seq_held = build_windows(held_out_df, feat_cols, tgt_cols,
                                                      W, H, SW, CONFIG["eval_stride"],
                                                      scaler, warmup)
        X_cross, y_cross_raw, seq_cross = build_windows(cross_df_full, feat_cols, tgt_cols,
                                                         W, H, SW, CONFIG["eval_stride"],
                                                         scaler, warmup)

        y_held_mean = y_held_raw.mean(axis=1)
        y_cross_mean = y_cross_raw.mean(axis=1)

        # Target normalization from training
        y_flat = y_train_raw.reshape(-1, N_TARGETS)
        t_mean = y_flat.mean(axis=0)
        t_std = y_flat.std(axis=0)
        t_std[t_std < 1e-8] = 1.0

        y_train_norm = (y_train_raw - t_mean) / t_std
        y_es_norm = (y_es_raw - t_mean) / t_std

        print(f"    Windows: train={len(y_train_raw)}, es={len(y_es_raw)}, "
              f"held_out={len(y_held_raw)}, cross={len(y_cross_raw)}", flush=True)

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        model, best_ep = train_model(X_train, y_train_norm, X_es, y_es_norm,
                                      n_feat, n_steps, device, fold_label)

        # Predict and denormalize
        pred_held_norm = predict(model, X_held, device)
        pred_cross_norm = predict(model, X_cross, device)
        pred_held = pred_held_norm * t_std + t_mean
        pred_cross = pred_cross_norm * t_std + t_mean

        elapsed = time.time() - t0

        held_metrics = compute_metrics(pred_held, y_held_mean, tgt_cols)
        cross_metrics = compute_metrics(pred_cross, y_cross_mean, tgt_cols)

        print(f"\n    Held-out camera ({held_out.split('_')[-1]}, n={len(y_held_mean)}):",
              flush=True)
        for name in tgt_cols:
            print(f"      {name:28s}: r={held_metrics[f'r_{name}']:.4f}  "
                  f"MSE={held_metrics[f'mse_{name}']:.6f}", flush=True)

        print(f"    Cross-camera (SE38th, n={len(y_cross_mean)}):", flush=True)
        for name in tgt_cols:
            print(f"      {name:28s}: r={cross_metrics[f'r_{name}']:.4f}  "
                  f"MSE={cross_metrics[f'mse_{name}']:.6f}", flush=True)

        print(f"    Best epoch: {best_ep}, time: {elapsed:.1f}s", flush=True)

        result = {
            "fold": fold_idx + 1,
            "held_out": held_out,
            "best_epoch": best_ep,
            "elapsed_s": round(elapsed, 1),
        }
        for k, v in held_metrics.items():
            result[f"heldout_{k}"] = round(v, 6)
        for k, v in cross_metrics.items():
            result[f"cross_{k}"] = round(v, 6)
        fold_results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary: mean ± std across folds
    print(f"\n{'='*80}")
    print(f"  4-FOLD SUMMARY (mean ± std)")
    print(f"{'='*80}")

    for split_prefix, label in [("heldout", "Held-out camera"), ("cross", "Cross (SE38th)")]:
        print(f"\n  {label}:", flush=True)
        for name in tgt_cols:
            r_key = f"{split_prefix}_r_{name}"
            mse_key = f"{split_prefix}_mse_{name}"
            r_vals = [r[r_key] for r in fold_results]
            mse_vals = [r[mse_key] for r in fold_results]
            print(f"    {name:28s}: r={np.mean(r_vals):.4f}±{np.std(r_vals):.4f}  "
                  f"MSE={np.mean(mse_vals):.6f}±{np.std(mse_vals):.6f}", flush=True)

    # Save TSV
    tsv_path = SCRIPT_DIR / "kfold_leave_one_camera_out.tsv"
    pd.DataFrame(fold_results).to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved {tsv_path}", flush=True)

    return fold_results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
    print(f"  Data loaded: {len(df)} rows", flush=True)

    exp1_results = experiment1(df, device)
    exp2_results = experiment2(df, device)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
