#!/usr/bin/env python3
"""
Subset validation: retrain clip_sm (h256 / L2) on top-K features (K ∈ {15,20,25})
from the permutation-importance combined_score ranking. 3 seeds each.

Compare against full 65-feature baseline:
  r_ns = 0.647 ± 0.016
  r_sm = 0.275 ± 0.019

Writes feature_ablation/subset_validation.tsv with per-subset mean ± std
for r_nano / r_recovery_ns / r_recovery_sm.
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
PERM_TSV = SCRIPT_DIR / "perm_importance_h256_full.tsv"

sys.path.insert(0, str(PROJECT_DIR / "second_autosearch" / "src"))
from pod_features import SPATIAL_65  # noqa: F401  (kept for reference)

CONFIG = {
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
SUBSET_SIZES = [15, 20, 25]

# Baseline for reference in final print
BASELINE = {"ns_mean": 0.647, "ns_std": 0.016, "sm_mean": 0.275, "sm_std": 0.019}


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
            print(f"      [{label}] ep{epoch:3d} val={vloss:.6f} best={best:.6f} pat={pat}",
                  flush=True)
        if pat >= CONFIG["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def evaluate(model, X_eval, y_eval_mean, device):
    with torch.no_grad():
        preds = []
        for s in range(0, len(X_eval), 512):
            xb = torch.from_numpy(X_eval[s:s + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred_mean = np.concatenate(preds, axis=0).mean(axis=1)
    return {
        "nano": float(np.corrcoef(pred_mean[:, 0], y_eval_mean[:, 0])[0, 1]),
        "ns":   float(np.corrcoef(pred_mean[:, 1], y_eval_mean[:, 1])[0, 1]),
        "sm":   float(np.corrcoef(pred_mean[:, 2], y_eval_mean[:, 2])[0, 1]),
    }


def run_subset(feat_cols, tgt_cols, df, device, label):
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

    n_feat = len(feat_cols)
    n_steps = CONFIG["horizon"] // CONFIG["sub_window"]

    per_seed = []
    for seed in SEEDS:
        print(f"\n    === {label} seed {seed} ===", flush=True)
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        t0 = time.time()
        model = train_one(X_train, y_train, X_es, y_es, n_feat, n_steps,
                          device, f"{label}_s{seed}")
        corrs = evaluate(model, X_eval, y_eval_mean, device)
        elapsed = time.time() - t0
        print(f"      nano r={corrs['nano']:.4f}  "
              f"rec_ns r={corrs['ns']:.4f}  "
              f"rec_sm r={corrs['sm']:.4f}  ({elapsed:.0f}s)",
              flush=True)
        per_seed.append(corrs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return per_seed


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    # Load ranking
    rank_df = pd.read_csv(PERM_TSV, sep="\t")
    rank_df = rank_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    ranked_features = rank_df["feature"].tolist()
    print(f"  Loaded ranking of {len(ranked_features)} features from {PERM_TSV.name}",
          flush=True)

    df = load_data()
    tgt_cols = CONFIG["targets"]

    results = {}  # k -> list of per-seed corr dicts
    for k in SUBSET_SIZES:
        feat_cols = ranked_features[:k]
        label = f"top{k}"
        print(f"\n  ═══ {label}: {k} features ═══", flush=True)
        print(f"    features: {feat_cols}", flush=True)
        results[k] = run_subset(feat_cols, tgt_cols, df, device, label)

    # Aggregate
    rows = []
    print(f"\n{'='*88}")
    print(f"  SUMMARY (3 seeds, mean ± std)")
    print(f"{'='*88}")
    print(f"  {'subset':10s} | {'nano (r)':>17s} | {'rec_ns (r)':>17s} | {'rec_sm (r)':>17s}")
    print(f"  {'-'*70}")
    print(f"  {'full_65':10s} | {'n/a':>17s} | "
          f"{BASELINE['ns_mean']:.4f} ± {BASELINE['ns_std']:.4f} | "
          f"{BASELINE['sm_mean']:.4f} ± {BASELINE['sm_std']:.4f}  (reference)")
    print(f"  {'-'*70}")
    for k in SUBSET_SIZES:
        per_seed = results[k]
        means, stds = {}, {}
        for key in ("nano", "ns", "sm"):
            vals = [c[key] for c in per_seed]
            means[key] = float(np.mean(vals))
            stds[key] = float(np.std(vals))
        print(f"  {f'top{k}':10s} | "
              f"{means['nano']:.4f} ± {stds['nano']:.4f} | "
              f"{means['ns']:.4f} ± {stds['ns']:.4f} | "
              f"{means['sm']:.4f} ± {stds['sm']:.4f}")
        # Deltas
        d_ns = means['ns'] - BASELINE['ns_mean']
        d_sm = means['sm'] - BASELINE['sm_mean']
        print(f"             Δ vs full_65:  "
              f"rec_ns={d_ns:+.4f}  rec_sm={d_sm:+.4f}")
        rows.append({
            "subset": f"top{k}",
            "n_features": k,
            "nano_r_mean": round(means["nano"], 4),
            "nano_r_std": round(stds["nano"], 4),
            "rec_ns_r_mean": round(means["ns"], 4),
            "rec_ns_r_std": round(stds["ns"], 4),
            "rec_sm_r_mean": round(means["sm"], 4),
            "rec_sm_r_std": round(stds["sm"], 4),
            "rec_ns_delta_vs_full65": round(means["ns"] - BASELINE["ns_mean"], 4),
            "rec_sm_delta_vs_full65": round(means["sm"] - BASELINE["sm_mean"], 4),
        })
    rows.append({
        "subset": "full_65",
        "n_features": 65,
        "nano_r_mean": None, "nano_r_std": None,
        "rec_ns_r_mean": BASELINE["ns_mean"], "rec_ns_r_std": BASELINE["ns_std"],
        "rec_sm_r_mean": BASELINE["sm_mean"], "rec_sm_r_std": BASELINE["sm_std"],
        "rec_ns_delta_vs_full65": 0.0, "rec_sm_delta_vs_full65": 0.0,
    })
    print(f"{'='*88}")

    out_df = pd.DataFrame(rows)
    tsv_path = SCRIPT_DIR / "subset_validation.tsv"
    out_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved {tsv_path}", flush=True)
    print(f"  Total time: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
