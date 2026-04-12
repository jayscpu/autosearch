#!/usr/bin/env python3
"""Feature selection over SPATIAL_95 (h128/L2, clip_sm_normed).

Step 1: Permutation importance on all 95 features (seed 42, 5 shuffles
        per feature). Score = r_ns drop + r_sm drop on within-camera.
Step 2: Subset sweep at {30,40,50,60,70,80,95} using the ranking (seed 42).
        Find sweet spot maximizing (within_r_ns + cross_r_nano).
Step 3: 3-seed validation (42,43,44) of the best subset.
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

CFG = {
    "targets": ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"],
    "window": 30, "horizon": 30, "sub_window": 6,
    "train_stride": 15, "eval_stride": 30, "warmup_frames": 400,
    "train_intersections": [
        "Bellevue_150th_Eastgate", "Bellevue_150th_Newport",
        "Bellevue_Bellevue_NE8th", "Bellevue_116th_NE12th",
    ],
    "test_intersection": "Bellevue_150th_SE38th",
    "train_fraction": 0.50, "earlystop_fraction": 0.60,
    "hidden_size": 128, "n_layers": 2, "dropout": 0.4,
    "lr": 1e-3, "weight_decay": 1e-5,
    "batch_size": 256, "max_epochs": 300, "patience": 40, "grad_clip": 1.0,
}
N_TARGETS = len(CFG["targets"])
N_SHUFFLES = 5
SUBSET_SIZES = [30, 40, 50, 60, 70, 80, 95]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["miss_rate_nano"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    df["miss_rate_small"] = df["fn_small"] / df["x_count"].clip(lower=1)
    df["miss_rate_medium"] = df["fn_medium"] / df["x_count"].clip(lower=1)
    df["recovery_nano_to_small"] = df["miss_rate_nano"] - df["miss_rate_small"]
    df["recovery_small_to_medium"] = (df["miss_rate_small"] - df["miss_rate_medium"]).clip(lower=0)
    return df


def split_intersection(df, name):
    sub = df[df["sequence"] == name].sort_values("frame_id")
    n = len(sub)
    t_end = int(n * CFG["train_fraction"])
    e_end = int(n * CFG["earlystop_fraction"])
    return sub.iloc[:t_end], sub.iloc[t_end:e_end], sub.iloc[e_end:]


def build_windows(df, feat_cols, tgt_cols, stride, scaler):
    W, H, SW = CFG["window"], CFG["horizon"], CFG["sub_window"]
    n_steps = H // SW
    total = W + H
    X_list, y_list, seq_labels = [], [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = CFG["warmup_frames"]
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
            X_list.append(fv[t:t + W])
            tg = np.zeros((n_steps, len(tgt_cols)), dtype=np.float32)
            for step in range(n_steps):
                s = t + W + step * SW
                tg[step] = tv[s:s + SW].mean(axis=0)
            y_list.append(tg)
            seq_labels.append(seq_name)
    if not X_list:
        return (np.zeros((0, W, len(feat_cols)), dtype=np.float32),
                np.zeros((0, n_steps, len(tgt_cols)), dtype=np.float32), [])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32), seq_labels)


def fit_scaler(train_df, feat_cols):
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[CFG["warmup_frames"]:])
    sc = StandardScaler()
    sc.fit(pd.concat(kept, ignore_index=True)[feat_cols].values)
    return sc


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
                            CFG["hidden_size"], CFG["n_layers"],
                            CFG["dropout"]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CFG["batch_size"], shuffle=True)
    vx, vy = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    opt = Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse = nn.MSELoss()
    best, best_state, best_ep, pat = float("inf"), None, 0, 0
    for epoch in range(CFG["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = mse(model(xb), yb)
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = mse(model(vx), vy).item()
        sched.step(vloss)
        if vloss < best:
            best = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = epoch
            pat = 0
        else:
            pat += 1
        if epoch % 20 == 0 or pat >= CFG["patience"]:
            print(f"    [{label}] ep{epoch:3d} val={vloss:.6f} best={best:.6f} pat={pat}",
                  flush=True)
        if pat >= CFG["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_ep


def predict(model, X, device):
    with torch.no_grad():
        out = []
        for s in range(0, len(X), 512):
            xb = torch.from_numpy(X[s:s + 512]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0).mean(axis=1)


def corr3(pred, true, tgt_cols):
    return {name: float(np.corrcoef(pred[:, i], true[:, i])[0, 1])
            for i, name in enumerate(tgt_cols)}


def metrics_with_mse(pred, true, tgt_cols):
    m = {}
    for i, name in enumerate(tgt_cols):
        m[f"r_{name}"] = float(np.corrcoef(pred[:, i], true[:, i])[0, 1])
        m[f"mse_{name}"] = float(np.mean((pred[:, i] - true[:, i]) ** 2))
    return m


def prepare(df, feat_cols):
    tgt_cols = CFG["targets"]
    train_dfs, es_dfs, within_dfs = [], [], []
    for name in CFG["train_intersections"]:
        tr, es, wv = split_intersection(df, name)
        train_dfs.append(tr); es_dfs.append(es); within_dfs.append(wv)
    train_df = pd.concat(train_dfs, ignore_index=True)
    es_df = pd.concat(es_dfs, ignore_index=True)
    within_df = pd.concat(within_dfs, ignore_index=True)
    cross_df = df[df["sequence"] == CFG["test_intersection"]].copy()

    scaler = fit_scaler(train_df, feat_cols)
    X_train, y_train_raw, _ = build_windows(train_df, feat_cols, tgt_cols,
                                             CFG["train_stride"], scaler)
    X_es, y_es_raw, _ = build_windows(es_df, feat_cols, tgt_cols,
                                       CFG["eval_stride"], scaler)
    X_w, y_w_raw, seq_w = build_windows(within_df, feat_cols, tgt_cols,
                                         CFG["eval_stride"], scaler)
    X_c, y_c_raw, seq_c = build_windows(cross_df, feat_cols, tgt_cols,
                                         CFG["eval_stride"], scaler)
    y_flat = y_train_raw.reshape(-1, N_TARGETS)
    t_mean = y_flat.mean(axis=0)
    t_std = y_flat.std(axis=0); t_std[t_std < 1e-8] = 1.0
    return dict(
        X_train=X_train, y_train_norm=(y_train_raw - t_mean) / t_std,
        X_es=X_es, y_es_norm=(y_es_raw - t_mean) / t_std,
        X_w=X_w, y_w_mean=y_w_raw.mean(axis=1), seq_w=seq_w,
        X_c=X_c, y_c_mean=y_c_raw.mean(axis=1), seq_c=seq_c,
        t_mean=t_mean, t_std=t_std,
        n_train=len(y_train_raw), n_es=len(y_es_raw),
    )


def set_seed(s):
    random.seed(s); torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Permutation importance
# ═══════════════════════════════════════════════════════════════════

def step1_perm_importance(df, device):
    print("\n" + "=" * 70, flush=True)
    print("  STEP 1: Permutation importance (seed 42, 5 shuffles)", flush=True)
    print("=" * 70, flush=True)
    feat_cols = SPATIAL_95
    tgt_cols = CFG["targets"]
    n_steps = CFG["horizon"] // CFG["sub_window"]
    n_feat = len(feat_cols)

    data = prepare(df, feat_cols)
    print(f"  Windows: train={data['n_train']} es={data['n_es']} "
          f"within={len(data['X_w'])} cross={len(data['X_c'])}", flush=True)

    set_seed(42)
    model, best_ep = train_model(
        data["X_train"], data["y_train_norm"],
        data["X_es"], data["y_es_norm"],
        n_feat, n_steps, device, "perm_base")
    print(f"  Base trained: best_ep={best_ep}", flush=True)

    # Baseline on within (denorm not required for corr)
    base_w = predict(model, data["X_w"], device) * data["t_std"] + data["t_mean"]
    base_c = corr3(base_w, data["y_w_mean"], tgt_cols)
    print(f"  Baseline within: r_nano={base_c[tgt_cols[0]]:.4f} "
          f"r_ns={base_c[tgt_cols[1]]:.4f} r_sm={base_c[tgt_cols[2]]:.4f}",
          flush=True)

    rng = np.random.default_rng(42)
    N, W, _ = data["X_w"].shape
    flat = N * W

    rows = []
    t0 = time.time()
    for fi, fname in enumerate(feat_cols):
        ns_drops, sm_drops = [], []
        for s in range(N_SHUFFLES):
            Xp = data["X_w"].copy()
            col = Xp[:, :, fi].reshape(flat)
            perm = rng.permutation(flat)
            Xp[:, :, fi] = col[perm].reshape(N, W)
            pp = predict(model, Xp, device) * data["t_std"] + data["t_mean"]
            pc = corr3(pp, data["y_w_mean"], tgt_cols)
            ns_drops.append(base_c[tgt_cols[1]] - pc[tgt_cols[1]])
            sm_drops.append(base_c[tgt_cols[2]] - pc[tgt_cols[2]])
        ns_m = float(np.mean(ns_drops))
        sm_m = float(np.mean(sm_drops))
        rows.append({"feature": fname, "ns_drop_mean": ns_m,
                     "sm_drop_mean": sm_m, "combined_score": ns_m + sm_m})
        if (fi + 1) % 10 == 0:
            el = time.time() - t0
            print(f"    {fi+1:3d}/{n_feat} [{el:.0f}s]", flush=True)

    ranked = sorted(rows, key=lambda r: -r["combined_score"])
    for rk, r in enumerate(ranked):
        r["rank"] = rk + 1

    out_df = pd.DataFrame([{
        "feature": r["feature"],
        "ns_drop_mean": round(r["ns_drop_mean"], 6),
        "sm_drop_mean": round(r["sm_drop_mean"], 6),
        "combined_score": round(r["combined_score"], 6),
        "rank": r["rank"],
    } for r in ranked])
    tsv = SCRIPT_DIR / "perm_importance_95.tsv"
    out_df.to_csv(tsv, sep="\t", index=False)
    print(f"\n  Saved {tsv}", flush=True)

    print("\n  Top 30:", flush=True)
    for r in ranked[:30]:
        tag = "NEW" if r["feature"] in NEW_FEATURES_30 else "   "
        print(f"    {r['rank']:3d}. [{tag}] {r['feature']:32s} "
              f"ns={r['ns_drop_mean']:+.4f} sm={r['sm_drop_mean']:+.4f} "
              f"combined={r['combined_score']:+.4f}", flush=True)

    print("\n  Bottom 30:", flush=True)
    for r in ranked[-30:]:
        tag = "NEW" if r["feature"] in NEW_FEATURES_30 else "   "
        print(f"    {r['rank']:3d}. [{tag}] {r['feature']:32s} "
              f"ns={r['ns_drop_mean']:+.4f} sm={r['sm_drop_mean']:+.4f} "
              f"combined={r['combined_score']:+.4f}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ranked


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Subset sweep
# ═══════════════════════════════════════════════════════════════════

def step2_subset_sweep(df, ranked, device):
    print("\n" + "=" * 70, flush=True)
    print("  STEP 2: Subset sweep (seed 42)", flush=True)
    print("=" * 70, flush=True)
    tgt_cols = CFG["targets"]
    n_steps = CFG["horizon"] // CFG["sub_window"]

    ordered_feats = [r["feature"] for r in ranked]
    sweep_rows = []

    for k in SUBSET_SIZES:
        feat_cols = ordered_feats[:k]
        print(f"\n  ── k={k} ──", flush=True)
        data = prepare(df, feat_cols)
        set_seed(42)
        model, best_ep = train_model(
            data["X_train"], data["y_train_norm"],
            data["X_es"], data["y_es_norm"],
            k, n_steps, device, f"k{k}")
        pred_w = predict(model, data["X_w"], device) * data["t_std"] + data["t_mean"]
        pred_c = predict(model, data["X_c"], device) * data["t_std"] + data["t_mean"]
        wm = metrics_with_mse(pred_w, data["y_w_mean"], tgt_cols)
        cm = metrics_with_mse(pred_c, data["y_c_mean"], tgt_cols)
        row = {"k": k, "best_epoch": best_ep}
        for name in tgt_cols:
            row[f"within_r_{name}"] = round(wm[f"r_{name}"], 6)
            row[f"cross_r_{name}"] = round(cm[f"r_{name}"], 6)
        sweep_rows.append(row)
        print(f"    within: nano={wm['r_miss_rate_nano']:.4f} "
              f"ns={wm['r_recovery_nano_to_small']:.4f} "
              f"sm={wm['r_recovery_small_to_medium']:.4f}", flush=True)
        print(f"    cross : nano={cm['r_miss_rate_nano']:.4f} "
              f"ns={cm['r_recovery_nano_to_small']:.4f} "
              f"sm={cm['r_recovery_small_to_medium']:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sw_df = pd.DataFrame(sweep_rows)
    tsv = SCRIPT_DIR / "subset_sweep.tsv"
    sw_df.to_csv(tsv, sep="\t", index=False)
    print(f"\n  Saved {tsv}", flush=True)

    # Sweet spot: maximize within_r_ns + cross_r_nano
    for r in sweep_rows:
        r["score"] = r["within_r_recovery_nano_to_small"] + r["cross_r_miss_rate_nano"]
    best = max(sweep_rows, key=lambda r: r["score"])
    print(f"\n  Sweep ranked by (within_r_ns + cross_r_nano):", flush=True)
    for r in sorted(sweep_rows, key=lambda x: -x["score"]):
        print(f"    k={r['k']:3d}  within_r_ns={r['within_r_recovery_nano_to_small']:.4f} "
              f"cross_r_nano={r['cross_r_miss_rate_nano']:.4f}  "
              f"score={r['score']:.4f}", flush=True)
    print(f"\n  >>> SWEET SPOT: k={best['k']} (score={best['score']:.4f})", flush=True)
    return best["k"], ordered_feats[:best["k"]]


# ═══════════════════════════════════════════════════════════════════
# STEP 3: 3-seed validation of best subset
# ═══════════════════════════════════════════════════════════════════

def step3_best_subset(df, best_feats, device):
    print("\n" + "=" * 70, flush=True)
    print(f"  STEP 3: 3-seed validation of best subset (k={len(best_feats)})",
          flush=True)
    print("=" * 70, flush=True)
    tgt_cols = CFG["targets"]
    n_steps = CFG["horizon"] // CFG["sub_window"]
    n_feat = len(best_feats)

    data = prepare(df, best_feats)
    results = []
    for seed in [42, 43, 44]:
        print(f"\n  ── Seed {seed} ──", flush=True)
        set_seed(seed)
        t0 = time.time()
        model, best_ep = train_model(
            data["X_train"], data["y_train_norm"],
            data["X_es"], data["y_es_norm"],
            n_feat, n_steps, device, f"best_seed{seed}")
        pw = predict(model, data["X_w"], device) * data["t_std"] + data["t_mean"]
        pc = predict(model, data["X_c"], device) * data["t_std"] + data["t_mean"]
        elapsed = time.time() - t0
        wm = metrics_with_mse(pw, data["y_w_mean"], tgt_cols)
        cm = metrics_with_mse(pc, data["y_c_mean"], tgt_cols)
        for split, m, n in [("within", wm, len(data["y_w_mean"])),
                             ("cross", cm, len(data["y_c_mean"]))]:
            print(f"    {split} (n={n}):", flush=True)
            for name in tgt_cols:
                print(f"      {name:28s}: r={m[f'r_{name}']:.4f} "
                      f"MSE={m[f'mse_{name}']:.6f}", flush=True)

        rows = []
        for i in range(len(data["y_w_mean"])):
            rows.append({"frame_idx": i, "split": "within",
                         "pred_nano": round(float(pw[i, 0]), 6),
                         "pred_recovery_ns": round(float(pw[i, 1]), 6),
                         "pred_recovery_sm": round(float(pw[i, 2]), 6),
                         "true_nano": round(float(data["y_w_mean"][i, 0]), 6),
                         "true_recovery_ns": round(float(data["y_w_mean"][i, 1]), 6),
                         "true_recovery_sm": round(float(data["y_w_mean"][i, 2]), 6),
                         "intersection": data["seq_w"][i]})
        for i in range(len(data["y_c_mean"])):
            rows.append({"frame_idx": i, "split": "cross",
                         "pred_nano": round(float(pc[i, 0]), 6),
                         "pred_recovery_ns": round(float(pc[i, 1]), 6),
                         "pred_recovery_sm": round(float(pc[i, 2]), 6),
                         "true_nano": round(float(data["y_c_mean"][i, 0]), 6),
                         "true_recovery_ns": round(float(data["y_c_mean"][i, 1]), 6),
                         "true_recovery_sm": round(float(data["y_c_mean"][i, 2]), 6),
                         "intersection": data["seq_c"][i] if i < len(data["seq_c"]) else CFG["test_intersection"]})
        csv = SCRIPT_DIR / f"predictions_best_seed{seed}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"    Saved {csv}", flush=True)

        result = {"seed": seed, "best_epoch": best_ep, "elapsed_s": round(elapsed, 1)}
        for k, v in wm.items(): result[f"within_{k}"] = round(v, 6)
        for k, v in cm.items(): result[f"cross_{k}"] = round(v, 6)
        results.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    summary = {"config": f"best_subset_k{len(best_feats)}_h128_L2"}
    for split in ["within", "cross"]:
        for name in tgt_cols:
            rs = [r[f"{split}_r_{name}"] for r in results]
            ms = [r[f"{split}_mse_{name}"] for r in results]
            summary[f"{split}_r_{name}_mean"] = round(np.mean(rs), 6)
            summary[f"{split}_r_{name}_std"] = round(np.std(rs), 6)
            summary[f"{split}_mse_{name}_mean"] = round(np.mean(ms), 6)
            summary[f"{split}_mse_{name}_std"] = round(np.std(ms), 6)

    tsv = SCRIPT_DIR / "best_subset_validation.tsv"
    pd.DataFrame(results).to_csv(tsv, sep="\t", index=False)
    with open(tsv, "a") as f:
        f.write(f"\n# Summary (mean±std): {summary}\n")
    print(f"\n  Saved {tsv}", flush=True)
    return summary


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════

def final_report(best_feats, best_summary):
    print("\n" + "=" * 70, flush=True)
    print("  FINAL COMPARISON", flush=True)
    print("=" * 70, flush=True)

    spatial65 = {
        "within_r_miss_rate_nano": (0.848, 0.003),
        "within_r_recovery_nano_to_small": (0.717, 0.024),
        "within_r_recovery_small_to_medium": (0.255, 0.044),
        "cross_r_miss_rate_nano": (0.449, 0.024),
        "cross_r_recovery_nano_to_small": (0.496, 0.048),
        "cross_r_recovery_small_to_medium": (0.330, 0.035),
    }
    spatial95 = {
        "within_r_miss_rate_nano": (0.861, 0.004),
        "within_r_recovery_nano_to_small": (0.701, 0.038),
        "within_r_recovery_small_to_medium": (0.200, 0.046),
        "cross_r_miss_rate_nano": (0.546, 0.017),
        "cross_r_recovery_nano_to_small": (0.525, 0.022),
        "cross_r_recovery_small_to_medium": (0.359, 0.011),
    }
    keys = [
        ("within_r_miss_rate_nano", "within_nano"),
        ("within_r_recovery_nano_to_small", "within_ns"),
        ("within_r_recovery_small_to_medium", "within_sm"),
        ("cross_r_miss_rate_nano", "cross_nano"),
        ("cross_r_recovery_nano_to_small", "cross_ns"),
        ("cross_r_recovery_small_to_medium", "cross_sm"),
    ]
    hdr = "Config       | " + " | ".join(f"{label:11s}" for _, label in keys)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for label, src in [("SPATIAL_65", spatial65), ("SPATIAL_95", spatial95)]:
        cells = [f"{src[k][0]:.3f}±{src[k][1]:.3f}" for k, _ in keys]
        print(f"{label:12s} | " + " | ".join(f"{c:11s}" for c in cells), flush=True)
    cells = [f"{best_summary[f'{k}_mean']:.3f}±{best_summary[f'{k}_std']:.3f}"
             for k, _ in keys]
    print(f"BEST_k{len(best_feats):<3d}   | " + " | ".join(f"{c:11s}" for c in cells),
          flush=True)

    dropped_from_65 = [f for f in SPATIAL_65 if f not in best_feats]
    new_kept = [f for f in NEW_FEATURES_30 if f in best_feats]
    print(f"\n  Best subset: k={len(best_feats)}", flush=True)
    print(f"\n  Dropped from SPATIAL_65 ({len(dropped_from_65)} features):",
          flush=True)
    for f in dropped_from_65:
        print(f"    - {f}", flush=True)
    print(f"\n  NEW features kept in best subset ({len(new_kept)}/30):",
          flush=True)
    for f in new_kept:
        print(f"    + {f}", flush=True)


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)
    df = load_data()
    print(f"  Data: {len(df)} rows", flush=True)

    ranked = step1_perm_importance(df, device)
    best_k, best_feats = step2_subset_sweep(df, ranked, device)
    best_summary = step3_best_subset(df, best_feats, device)
    final_report(best_feats, best_summary)

    print(f"\n  Total time: {(time.time() - t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
