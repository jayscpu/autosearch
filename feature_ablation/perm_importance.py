#!/usr/bin/env python3
"""
Permutation importance on clip_sm model (h256 / L2) across 3 seeds.

For each of the 65 SPATIAL_65 features, shuffle that feature's values in the
eval set (10 shuffles per feature per seed) and measure the drop in
r_recovery_ns (recovery_nano_to_small) and r_recovery_sm (recovery_small_to_medium)
correlation vs. the baseline (un-shuffled) predictions.

Outputs
  feature_ablation/perm_importance_h256.tsv  (per-feature drops)
  feature_ablation/perm_importance_h256.png  (three bar charts)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "second_autosearch" / "data"
FEATURES_CSV = DATA_DIR / "pod_features_all.csv"

sys.path.insert(0, str(PROJECT_DIR / "second_autosearch" / "src"))
from pod_features import SPATIAL_65

# ── Config (clip_sm, h256 / L2) ──
CONFIG = {
    "features": SPATIAL_65,
    "targets": ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"],
    "window": 30, "horizon": 30, "sub_window": 6,
    "train_stride": 15, "eval_stride": 30, "warmup_frames": 400,
    "train_intersections": [
        "Bellevue_150th_Eastgate",
        "Bellevue_150th_Newport",
        "Bellevue_Bellevue_NE8th",
        "Bellevue_116th_NE12th",
    ],
    "test_intersection": "Bellevue_150th_SE38th",
    "train_fraction": 0.50, "earlystop_fraction": 0.60,
    "hidden_size": 256, "n_layers": 2, "dropout": 0.4,
    "lr": 1e-3, "weight_decay": 1e-5,
    "batch_size": 256, "max_epochs": 300, "patience": 40, "grad_clip": 1.0,
}
N_TARGETS = len(CONFIG["targets"])
SEEDS = [42, 43, 44]
N_SHUFFLES = 10


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


def split_data(df):
    train_dfs, es_dfs, within_dfs = [], [], []
    for name in CONFIG["train_intersections"]:
        sub = df[df["sequence"] == name].sort_values("frame_id")
        n = len(sub)
        t_end = int(n * CONFIG["train_fraction"])
        e_end = int(n * CONFIG["earlystop_fraction"])
        train_dfs.append(sub.iloc[:t_end])
        es_dfs.append(sub.iloc[t_end:e_end])
        within_dfs.append(sub.iloc[e_end:])
    return (pd.concat(train_dfs, ignore_index=True),
            pd.concat(es_dfs, ignore_index=True),
            pd.concat(within_dfs, ignore_index=True),
            df[df["sequence"] == CONFIG["test_intersection"]].copy())


def build_windows(df, feat_cols, tgt_cols, stride, scaler):
    W, H, S = CONFIG["window"], CONFIG["horizon"], CONFIG["sub_window"]
    n_steps = H // S
    total = W + H
    X_list, y_list = [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = CONFIG["warmup_frames"]
    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        if len(sdf) < total:
            continue
        fv = scaler.transform(sdf[feat_cols].values)
        tv = sdf[tgt_cols].values
        for t in range(0, len(sdf) - total + 1, stride):
            X_list.append(fv[t:t + W])
            tg = np.zeros((n_steps, len(tgt_cols)), dtype=np.float32)
            for step in range(n_steps):
                s = t + W + step * S
                tg[step] = tv[s:s + S].mean(axis=0)
            y_list.append(tg)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


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


def train_one(X_train, y_train, X_val, y_val, n_feat, n_steps, device, label):
    model = MultiOutputLSTM(n_feat, n_steps, N_TARGETS,
                            CONFIG["hidden_size"], CONFIG["n_layers"],
                            CONFIG["dropout"]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"], shuffle=True)
    vx, vy = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    opt = Adam(model.parameters(), lr=CONFIG["lr"],
               weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse = nn.MSELoss()
    best, best_state, pat = float("inf"), None, 0
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
            pat = 0
        else:
            pat += 1
        if epoch % 20 == 0 or pat >= CONFIG["patience"]:
            print(f"    [{label}] ep{epoch:3d} val={vloss:.6f} best={best:.6f} pat={pat}",
                  flush=True)
        if pat >= CONFIG["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def predict(model, X_eval, device):
    with torch.no_grad():
        out = []
        for s in range(0, len(X_eval), 512):
            xb = torch.from_numpy(X_eval[s:s + 512]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0).mean(axis=1)  # (N, N_TARGETS)


def corrs_from_preds(pred_mean, y_eval_mean):
    return {
        "nano": float(np.corrcoef(pred_mean[:, 0], y_eval_mean[:, 0])[0, 1]),
        "ns": float(np.corrcoef(pred_mean[:, 1], y_eval_mean[:, 1])[0, 1]),
        "sm": float(np.corrcoef(pred_mean[:, 2], y_eval_mean[:, 2])[0, 1]),
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
    feat_cols = CONFIG["features"]
    tgt_cols = CONFIG["targets"]
    n_feat = len(feat_cols)
    n_steps = CONFIG["horizon"] // CONFIG["sub_window"]
    print(f"  n_features={n_feat}, n_steps={n_steps}", flush=True)

    train_df, es_df, within_df, cross_df = split_data(df)

    warmup = CONFIG["warmup_frames"]
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[warmup:])
    scaler = StandardScaler()
    scaler.fit(pd.concat(kept, ignore_index=True)[feat_cols].values)

    X_train, y_train = build_windows(train_df, feat_cols, tgt_cols,
                                      CONFIG["train_stride"], scaler)
    X_es, y_es = build_windows(es_df, feat_cols, tgt_cols,
                                CONFIG["eval_stride"], scaler)
    X_w, y_w = build_windows(within_df, feat_cols, tgt_cols,
                              CONFIG["eval_stride"], scaler)
    X_c, y_c = build_windows(cross_df, feat_cols, tgt_cols,
                              CONFIG["eval_stride"], scaler)

    X_eval = np.concatenate([X_w, X_c], axis=0)
    y_eval = np.concatenate([y_w, y_c], axis=0)
    y_eval_mean = y_eval.mean(axis=1)
    print(f"  Windows: train={len(y_train)} es={len(y_es)} eval={len(X_eval)}",
          flush=True)

    # drops[seed_idx][feat_idx] -> 10 per-shuffle drops, per target
    all_nano_drops = np.zeros((len(SEEDS), n_feat, N_SHUFFLES), dtype=np.float64)
    all_ns_drops = np.zeros((len(SEEDS), n_feat, N_SHUFFLES), dtype=np.float64)
    all_sm_drops = np.zeros((len(SEEDS), n_feat, N_SHUFFLES), dtype=np.float64)

    weights_dir = SCRIPT_DIR / "weights"
    weights_dir.mkdir(exist_ok=True)

    for si, seed in enumerate(SEEDS):
        print(f"\n  ═══ Seed {seed} ═══", flush=True)
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        weight_path = weights_dir / f"clip_sm_h256_L2_seed{seed}.pt"
        if weight_path.exists():
            print(f"    loading cached weights: {weight_path.name}", flush=True)
            model = MultiOutputLSTM(n_feat, n_steps, N_TARGETS,
                                    CONFIG["hidden_size"], CONFIG["n_layers"],
                                    CONFIG["dropout"]).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
        else:
            model = train_one(X_train, y_train, X_es, y_es, n_feat, n_steps,
                              device, f"seed{seed}")
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       weight_path)
            print(f"    saved weights: {weight_path.name}", flush=True)

        # Baseline predictions / correlations
        base_pred = predict(model, X_eval, device)
        base = corrs_from_preds(base_pred, y_eval_mean)
        print(f"    baseline: r_nano={base['nano']:.4f}  "
              f"r_ns={base['ns']:.4f}  r_sm={base['sm']:.4f}",
              flush=True)

        # Feature-level permutation importance
        # X_eval shape: (N, W, n_feat). Shuffle the column for feature f
        # across all (sample, time) positions jointly so cross-time structure
        # is destroyed while marginal distribution is preserved.
        rng = np.random.default_rng(seed)
        N, W, _ = X_eval.shape
        flat_len = N * W

        feat_t0 = time.time()
        for fi in range(n_feat):
            for si_shuf in range(N_SHUFFLES):
                X_perm = X_eval.copy()
                col = X_perm[:, :, fi].reshape(flat_len)
                perm = rng.permutation(flat_len)
                X_perm[:, :, fi] = col[perm].reshape(N, W)
                perm_pred = predict(model, X_perm, device)
                perm_c = corrs_from_preds(perm_pred, y_eval_mean)
                all_nano_drops[si, fi, si_shuf] = base["nano"] - perm_c["nano"]
                all_ns_drops[si, fi, si_shuf] = base["ns"] - perm_c["ns"]
                all_sm_drops[si, fi, si_shuf] = base["sm"] - perm_c["sm"]
            if (fi + 1) % 10 == 0 or fi == n_feat - 1:
                el = time.time() - feat_t0
                mean_nano = all_nano_drops[si, fi].mean()
                mean_ns = all_ns_drops[si, fi].mean()
                mean_sm = all_sm_drops[si, fi].mean()
                print(f"    feat {fi+1:2d}/{n_feat} ({feat_cols[fi]:30s}) "
                      f"nano={mean_nano:+.4f} ns={mean_ns:+.4f} sm={mean_sm:+.4f}  "
                      f"[{el:.0f}s]", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate across seeds × shuffles (3 * 10 = 30 samples per feature)
    rows = []
    for fi, fname in enumerate(feat_cols):
        nano_vals = all_nano_drops[:, fi, :].reshape(-1)
        ns_vals = all_ns_drops[:, fi, :].reshape(-1)
        sm_vals = all_sm_drops[:, fi, :].reshape(-1)
        nano_mean, nano_std = float(nano_vals.mean()), float(nano_vals.std())
        ns_mean, ns_std = float(ns_vals.mean()), float(ns_vals.std())
        sm_mean, sm_std = float(sm_vals.mean()), float(sm_vals.std())
        combined = 0.25 * nano_mean + 0.375 * ns_mean + 0.375 * sm_mean
        rows.append({
            "feature": fname,
            "nano_drop_mean": round(nano_mean, 6),
            "nano_drop_std": round(nano_std, 6),
            "ns_drop_mean": round(ns_mean, 6),
            "ns_drop_std": round(ns_std, 6),
            "sm_drop_mean": round(sm_mean, 6),
            "sm_drop_std": round(sm_std, 6),
            "combined_score": round(combined, 6),
        })

    out_df = pd.DataFrame(rows)
    tsv_path = SCRIPT_DIR / "perm_importance_h256_full.tsv"
    out_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved {tsv_path}", flush=True)

    print(f"\n  Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
