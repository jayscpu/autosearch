#!/usr/bin/env python3
"""
Architecture sweep for clip_sm config (3-seed averaging).
Configs: h64/L2, h128/L2, h128/L4, h256/L2, h256/L4 (all d0.4).
Saves results to arch_check/arch_sweep.tsv.
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

BASE_CONFIG = {
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
    "dropout": 0.4,
    "lr": 1e-3, "weight_decay": 1e-5,
    "batch_size": 256, "max_epochs": 300, "patience": 40, "grad_clip": 1.0,
}
N_TARGETS = len(BASE_CONFIG["targets"])
SEEDS = [42, 43, 44]

ARCHS = [
    (64, 2),
    (128, 2),
    (128, 4),
    (256, 2),
    (256, 4),
]


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
    for int_name in BASE_CONFIG["train_intersections"]:
        sub = df[df["sequence"] == int_name].sort_values("frame_id")
        n = len(sub)
        t_end = int(n * BASE_CONFIG["train_fraction"])
        e_end = int(n * BASE_CONFIG["earlystop_fraction"])
        train_dfs.append(sub.iloc[:t_end])
        es_dfs.append(sub.iloc[t_end:e_end])
        within_dfs.append(sub.iloc[e_end:])
    return (pd.concat(train_dfs, ignore_index=True),
            pd.concat(es_dfs, ignore_index=True),
            pd.concat(within_dfs, ignore_index=True),
            df[df["sequence"] == BASE_CONFIG["test_intersection"]].copy())


def build_windows(df, feat_cols, tgt_cols, stride, scaler):
    W, H, S = BASE_CONFIG["window"], BASE_CONFIG["horizon"], BASE_CONFIG["sub_window"]
    n_steps = H // S
    total = W + H
    X_list, y_list = [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    warmup = BASE_CONFIG["warmup_frames"]
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


def train_one(X_train, y_train, X_val, y_val, n_feat, n_steps,
              hidden, n_layers, device, label):
    model = MultiOutputLSTM(n_feat, n_steps, N_TARGETS, hidden, n_layers,
                            BASE_CONFIG["dropout"]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BASE_CONFIG["batch_size"], shuffle=True)
    vx, vy = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    opt = Adam(model.parameters(), lr=BASE_CONFIG["lr"],
               weight_decay=BASE_CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    mse = nn.MSELoss()
    best, best_state, pat = float("inf"), None, 0
    for epoch in range(BASE_CONFIG["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = mse(model(xb), yb)
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), BASE_CONFIG["grad_clip"])
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
        if epoch % 20 == 0 or pat >= BASE_CONFIG["patience"]:
            print(f"    [{label}] ep{epoch:3d} val={vloss:.6f} best={best:.6f} pat={pat}",
                  flush=True)
        if pat >= BASE_CONFIG["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def evaluate(model, X_eval, y_eval_mean, device, target_cols):
    with torch.no_grad():
        preds = []
        for s in range(0, len(X_eval), 512):
            xb = torch.from_numpy(X_eval[s:s + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred_mean = np.concatenate(preds, axis=0).mean(axis=1)
    corrs = {}
    for i, name in enumerate(target_cols):
        corrs[name] = float(np.corrcoef(pred_mean[:, i], y_eval_mean[:, i])[0, 1])
    return corrs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
    feat_cols = BASE_CONFIG["features"]
    tgt_cols = BASE_CONFIG["targets"]
    n_steps = BASE_CONFIG["horizon"] // BASE_CONFIG["sub_window"]

    train_df, es_df, within_df, cross_df = split_data(df)

    warmup = BASE_CONFIG["warmup_frames"]
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[warmup:])
    scaler = StandardScaler()
    scaler.fit(pd.concat(kept, ignore_index=True)[feat_cols].values)

    X_train, y_train = build_windows(train_df, feat_cols, tgt_cols,
                                      BASE_CONFIG["train_stride"], scaler)
    X_es, y_es = build_windows(es_df, feat_cols, tgt_cols,
                                BASE_CONFIG["eval_stride"], scaler)
    X_w, y_w = build_windows(within_df, feat_cols, tgt_cols,
                              BASE_CONFIG["eval_stride"], scaler)
    X_c, y_c = build_windows(cross_df, feat_cols, tgt_cols,
                              BASE_CONFIG["eval_stride"], scaler)

    X_eval = np.concatenate([X_w, X_c], axis=0)
    y_eval = np.concatenate([y_w, y_c], axis=0)
    y_eval_mean = y_eval.mean(axis=1)
    n_feat = len(feat_cols)
    print(f"  Windows: train={len(y_train)} es={len(y_es)} eval={len(y_eval)}",
          flush=True)

    # results[arch_label] -> list of per-seed corr dicts
    results = {}

    for hidden, n_layers in ARCHS:
        arch_label = f"h{hidden}_L{n_layers}"
        results[arch_label] = []
        for seed in SEEDS:
            label = f"{arch_label}_s{seed}"
            print(f"\n  === {label} ===", flush=True)
            random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            t0 = time.time()
            model = train_one(X_train, y_train, X_es, y_es, n_feat, n_steps,
                              hidden, n_layers, device, label)
            corrs = evaluate(model, X_eval, y_eval_mean, device, tgt_cols)
            elapsed = time.time() - t0
            print(f"    nano r={corrs['miss_rate_nano']:.4f}  "
                  f"rec_ns r={corrs['recovery_nano_to_small']:.4f}  "
                  f"rec_sm r={corrs['recovery_small_to_medium']:.4f}  "
                  f"({elapsed:.0f}s)", flush=True)
            results[arch_label].append(corrs)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Per-seed table ──
    print(f"\n{'='*100}")
    print(f"  PER-SEED CORRELATIONS")
    print(f"{'='*100}")
    print(f"  {'Arch':12s} {'Seed':>5s} | {'nano':>8s} | {'rec_ns':>8s} | {'rec_sm':>8s}")
    print(f"  {'-'*54}")
    for arch_label in results:
        for seed, corrs in zip(SEEDS, results[arch_label]):
            print(f"  {arch_label:12s} {seed:>5d} | "
                  f"{corrs['miss_rate_nano']:>8.4f} | "
                  f"{corrs['recovery_nano_to_small']:>8.4f} | "
                  f"{corrs['recovery_small_to_medium']:>8.4f}")
        print(f"  {'-'*54}")

    # ── Mean ± std summary ──
    print(f"\n{'='*100}")
    print(f"  MEAN ± STD (3 seeds)")
    print(f"{'='*100}")
    print(f"  {'Arch':12s} | {'nano (r)':>17s} | {'rec_ns (r)':>17s} | {'rec_sm (r)':>17s}")
    print(f"  {'-'*76}")

    summary_rows = []
    for arch_label in results:
        seeds_corrs = results[arch_label]
        means, stds = {}, {}
        for t in tgt_cols:
            vals = [c[t] for c in seeds_corrs]
            means[t] = float(np.mean(vals))
            stds[t] = float(np.std(vals))
        print(f"  {arch_label:12s} | "
              f"{means['miss_rate_nano']:.4f} ± {stds['miss_rate_nano']:.4f} | "
              f"{means['recovery_nano_to_small']:.4f} ± {stds['recovery_nano_to_small']:.4f} | "
              f"{means['recovery_small_to_medium']:.4f} ± {stds['recovery_small_to_medium']:.4f}")
        summary_rows.append({
            "arch": arch_label,
            "nano_r_mean": round(means["miss_rate_nano"], 4),
            "nano_r_std": round(stds["miss_rate_nano"], 4),
            "rec_ns_r_mean": round(means["recovery_nano_to_small"], 4),
            "rec_ns_r_std": round(stds["recovery_nano_to_small"], 4),
            "rec_sm_r_mean": round(means["recovery_small_to_medium"], 4),
            "rec_sm_r_std": round(stds["recovery_small_to_medium"], 4),
        })
    print(f"{'='*100}")

    tsv_path = SCRIPT_DIR / "arch_sweep.tsv"
    pd.DataFrame(summary_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Summary saved: {tsv_path}", flush=True)


if __name__ == "__main__":
    main()
