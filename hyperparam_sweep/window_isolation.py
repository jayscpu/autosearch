#!/usr/bin/env python3
"""
Window isolation sweep: 4 configs varying only window size.
Base: clip_sm h256/L2/d0.4, horizon=30, lr=3e-3, sub_window=6, 65 features, seed 42.
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

BASE = {
    "features": SPATIAL_65,
    "targets": ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"],
    "sub_window": 6,
    "horizon": 30,
    "lr": 3e-3,
    "dropout": 0.4,
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
    "n_layers": 2,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 300,
    "patience": 40,
    "grad_clip": 1.0,
    "seed": 42,
}

N_TARGETS = len(BASE["targets"])
WINDOWS = [15, 20, 30, 40]
TSV_PATH = SCRIPT_DIR / "window_isolation.tsv"
TSV_COLUMNS = [
    "config_id", "window", "horizon", "lr", "dropout",
    "r_nano", "r_ns", "r_sm", "combined_score",
]


def _fsync_write(path, mode, content):
    with open(path, mode) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


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
    for name in BASE["train_intersections"]:
        sub = df[df["sequence"] == name].sort_values("frame_id")
        n = len(sub)
        t_end = int(n * BASE["train_fraction"])
        e_end = int(n * BASE["earlystop_fraction"])
        train_dfs.append(sub.iloc[:t_end])
        es_dfs.append(sub.iloc[t_end:e_end])
        within_dfs.append(sub.iloc[e_end:])
    return (pd.concat(train_dfs, ignore_index=True),
            pd.concat(es_dfs, ignore_index=True),
            pd.concat(within_dfs, ignore_index=True),
            df[df["sequence"] == BASE["test_intersection"]].copy())


def build_windows(df, feat_cols, tgt_cols, window, horizon, sub_window,
                  stride, scaler):
    n_steps = horizon // sub_window
    total = window + horizon
    X_list, y_list = [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = BASE["warmup_frames"]
    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        if len(sdf) < total:
            continue
        fv = scaler.transform(sdf[feat_cols].values)
        tv = sdf[tgt_cols].values
        for t in range(0, len(sdf) - total + 1, stride):
            X_list.append(fv[t:t + window])
            tg = np.zeros((n_steps, len(tgt_cols)), dtype=np.float32)
            for step in range(n_steps):
                s = t + window + step * sub_window
                tg[step] = tv[s:s + sub_window].mean(axis=0)
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


def train_one(X_train, y_train, X_val, y_val, n_feat, n_steps,
              lr, dropout, device, label):
    model = MultiOutputLSTM(n_feat, n_steps, N_TARGETS,
                            BASE["hidden_size"], BASE["n_layers"],
                            dropout).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BASE["batch_size"], shuffle=True)
    vx, vy = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=BASE["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse = nn.MSELoss()
    best, best_state, best_epoch, pat = float("inf"), None, 0, 0
    for epoch in range(BASE["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = mse(model(xb), yb)
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), BASE["grad_clip"])
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
        if epoch % 10 == 0 or pat >= BASE["patience"]:
            print(f"    {label} ep{epoch}: val={vloss:.6f} best={best:.6f} pat={pat}",
                  flush=True)
        if pat >= BASE["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best, best_epoch


def evaluate(model, X_eval, y_eval_mean, device):
    with torch.no_grad():
        preds = []
        for s in range(0, len(X_eval), 512):
            xb = torch.from_numpy(X_eval[s:s + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    pm = np.concatenate(preds, axis=0).mean(axis=1)
    return {
        "nano": float(np.corrcoef(pm[:, 0], y_eval_mean[:, 0])[0, 1]),
        "ns":   float(np.corrcoef(pm[:, 1], y_eval_mean[:, 1])[0, 1]),
        "sm":   float(np.corrcoef(pm[:, 2], y_eval_mean[:, 2])[0, 1]),
    }


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
    feat_cols = BASE["features"]
    tgt_cols = BASE["targets"]
    n_feat = len(feat_cols)
    sub_window = BASE["sub_window"]
    horizon = BASE["horizon"]
    n_steps = horizon // sub_window
    lr = BASE["lr"]
    dropout = BASE["dropout"]

    train_df, es_df, within_df, cross_df = split_data(df)

    # Fit scaler on warmup-skipped training data
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[BASE["warmup_frames"]:])
    scaler = StandardScaler()
    scaler.fit(pd.concat(kept, ignore_index=True)[feat_cols].values)

    # Resume support
    done_ids = set()
    if TSV_PATH.exists():
        try:
            existing = pd.read_csv(TSV_PATH, sep="\t")
            done_ids = set(existing["config_id"].tolist())
            print(f"  Resuming: {len(done_ids)} configs already done", flush=True)
        except Exception:
            pass
    else:
        _fsync_write(TSV_PATH, "w", "\t".join(TSV_COLUMNS) + "\n")

    results = []

    for window in WINDOWS:
        cid = f"w{window}_h{horizon}_lr{lr:g}_d{dropout}"
        if cid in done_ids:
            print(f"  Skipping {cid} (already done)", flush=True)
            continue

        train_stride = max(1, window // 2)
        eval_stride = window

        print(f"\n  Building windows for window={window} ...", flush=True)
        X_train, y_train = build_windows(train_df, feat_cols, tgt_cols,
                                          window, horizon, sub_window,
                                          train_stride, scaler)
        X_es, y_es = build_windows(es_df, feat_cols, tgt_cols,
                                    window, horizon, sub_window,
                                    eval_stride, scaler)
        X_w, y_w = build_windows(within_df, feat_cols, tgt_cols,
                                  window, horizon, sub_window,
                                  eval_stride, scaler)
        X_c, y_c = build_windows(cross_df, feat_cols, tgt_cols,
                                  window, horizon, sub_window,
                                  eval_stride, scaler)
        X_eval = np.concatenate([X_w, X_c], axis=0)
        y_eval = np.concatenate([y_w, y_c], axis=0)
        y_eval_mean = y_eval.mean(axis=1)

        print(f"  {cid}: train={len(y_train)} es={len(y_es)} eval={len(y_eval)}",
              flush=True)

        seed = BASE["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        model, best_val, best_epoch = train_one(
            X_train, y_train, X_es, y_es, n_feat, n_steps,
            lr, dropout, device, cid)
        corrs = evaluate(model, X_eval, y_eval_mean, device)
        combined = 0.5 * corrs["ns"] + 0.5 * corrs["sm"]
        elapsed = time.time() - t0

        print(f"  => {cid}: nano={corrs['nano']:.4f} ns={corrs['ns']:.4f} "
              f"sm={corrs['sm']:.4f} comb={combined:.4f} ep{best_epoch} ({elapsed:.0f}s)",
              flush=True)

        row = {
            "config_id": cid, "window": window, "horizon": horizon,
            "lr": lr, "dropout": dropout,
            "r_nano": round(corrs["nano"], 6),
            "r_ns": round(corrs["ns"], 6),
            "r_sm": round(corrs["sm"], 6),
            "combined_score": round(combined, 6),
        }
        results.append(row)
        _fsync_write(TSV_PATH, "a",
                     "\t".join(str(row[c]) for c in TSV_COLUMNS) + "\n")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print final results table
    print(f"\n{'='*80}")
    print(f"  Window Isolation Results (h256/L2/d0.4, horizon=30, lr=3e-3)")
    print(f"{'='*80}")

    all_df = pd.read_csv(TSV_PATH, sep="\t")
    print(f"\n  {'config_id':32s} {'r_nano':>8s} {'r_ns':>8s} {'r_sm':>8s} {'combined':>10s}")
    print(f"  {'-'*66}")
    for _, r in all_df.iterrows():
        print(f"  {r['config_id']:32s} {r['r_nano']:>8.4f} {r['r_ns']:>8.4f} "
              f"{r['r_sm']:>8.4f} {r['combined_score']:>10.4f}")

    best_row = all_df.loc[all_df["combined_score"].idxmax()]
    print(f"\n  Best: {best_row['config_id']} (combined={best_row['combined_score']:.4f})")

    elapsed_total = time.time() - t_start
    print(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
