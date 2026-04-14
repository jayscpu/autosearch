#!/usr/bin/env python3
"""Part 1: Train k=80 model (3 seeds) and save full per-step predictions.

Same protocol as new_features/feature_selection.py step 3, but the 5 decoder
steps are preserved instead of averaged. Saves refinedsys/predictions_multistep_seed{42,43,44}.csv.
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
FEATURES_CSV = PROJECT_DIR / "new_features" / "pod_features_merged_95.csv"
RANK_TSV = PROJECT_DIR / "new_features" / "perm_importance_95.tsv"
K = 80

TARGETS = ["miss_rate_nano", "recovery_nano_to_small", "recovery_small_to_medium"]
N_TARGETS = 3
SEEDS = [42, 43, 44]

CFG = {
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


def load_feats():
    imp = pd.read_csv(RANK_TSV, sep="\t")
    feats = imp.sort_values("rank").head(K)["feature"].tolist()
    assert len(feats) == K
    return feats


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
    t = int(n * CFG["train_fraction"]); e = int(n * CFG["earlystop_fraction"])
    return sub.iloc[:t], sub.iloc[t:e], sub.iloc[e:]


def build_windows(df, feat_cols, tgt_cols, stride, scaler):
    W, H, SW = CFG["window"], CFG["horizon"], CFG["sub_window"]
    n_steps = H // SW
    total = W + H
    warmup = CFG["warmup_frames"]
    X, Y, seq_labels = [], [], []
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
            X.append(fv[t:t + W])
            tg = np.zeros((n_steps, len(tgt_cols)), dtype=np.float32)
            for step in range(n_steps):
                s = t + W + step * SW
                tg[step] = tv[s:s + SW].mean(axis=0)
            Y.append(tg)
            seq_labels.append(seq_name)
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32), seq_labels)


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
            best_ep = epoch; pat = 0
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


def predict_all_steps(model, X, device):
    """Return (N, n_steps, n_targets) WITHOUT averaging."""
    with torch.no_grad():
        out = []
        for s in range(0, len(X), 512):
            xb = torch.from_numpy(X[s:s + 512]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def prepare(df, feat_cols):
    tgt_cols = TARGETS
    train_dfs, es_dfs, within_dfs = [], [], []
    for intx in CFG["train_intersections"]:
        tr, es, wv = split_intersection(df, intx)
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
        X_w=X_w, y_w_raw=y_w_raw, seq_w=seq_w,
        X_c=X_c, y_c_raw=y_c_raw, seq_c=seq_c,
        t_mean=t_mean, t_std=t_std,
    )


def save_predictions(seed, preds_w_real, preds_c_real,
                     y_w_raw, y_c_raw, seq_w, seq_c):
    """preds_w/c: (N, 5, 3) denormalized."""
    n_steps = preds_w_real.shape[1]
    assert n_steps == 5
    rows = []
    target_short = {0: "nano", 1: "recovery_ns", 2: "recovery_sm"}

    def emit(i, split, preds, y_raw, seq):
        row = {"frame_idx": i, "split": split,
                "intersection": seq[i] if i < len(seq) else CFG["test_intersection"]}
        for t in range(3):
            for s in range(n_steps):
                row[f"pred_{target_short[t]}_s{s}"] = round(float(preds[i, s, t]), 6)
            row[f"pred_{target_short[t]}_avg"] = round(float(preds[i, :, t].mean()), 6)
            for s in range(n_steps):
                row[f"true_{target_short[t]}_s{s}"] = round(float(y_raw[i, s, t]), 6)
        return row

    for i in range(len(preds_w_real)):
        rows.append(emit(i, "within", preds_w_real, y_w_raw, seq_w))
    for i in range(len(preds_c_real)):
        rows.append(emit(i, "cross", preds_c_real, y_c_raw, seq_c))

    out = pd.DataFrame(rows)
    path = SCRIPT_DIR / f"predictions_multistep_seed{seed}.csv"
    out.to_csv(path, index=False)
    return path


def verify_avg_matches_baseline(seed):
    """Spot check: avg of per-step preds matches predictions_best_seed{seed}.csv."""
    base_path = PROJECT_DIR / "new_features" / f"predictions_best_seed{seed}.csv"
    if not base_path.exists():
        print(f"    (skip verify: {base_path} not found)", flush=True)
        return
    new_path = SCRIPT_DIR / f"predictions_multistep_seed{seed}.csv"
    b = pd.read_csv(base_path)
    n = pd.read_csv(new_path)
    # Compare pred_*_avg with baseline pred_nano / pred_recovery_ns / pred_recovery_sm.
    # Note: the two runs use independent model trainings with the same seed, but
    # identical code paths / hyperparams / features → should match bitwise.
    for col_new, col_base in [("pred_nano_avg", "pred_nano"),
                               ("pred_recovery_ns_avg", "pred_recovery_ns"),
                               ("pred_recovery_sm_avg", "pred_recovery_sm")]:
        a = n[col_new].values; b_ = b[col_base].values
        if len(a) != len(b_):
            print(f"    ! {col_new}: length mismatch ({len(a)} vs {len(b_)})",
                  flush=True); return
        diff = np.max(np.abs(a - b_))
        print(f"    verify {col_new} vs baseline {col_base}: max|Δ|={diff:.2e}",
              flush=True)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)
    feats = load_feats()
    print(f"  Features: {len(feats)} (top-{K})", flush=True)
    df = load_data()
    data = prepare(df, feats)
    print(f"  Windows: train={len(data['X_train'])} es={len(data['X_es'])} "
          f"within={len(data['X_w'])} cross={len(data['X_c'])}", flush=True)

    n_feat = len(feats)
    n_steps = CFG["horizon"] // CFG["sub_window"]

    for seed in SEEDS:
        print(f"\n  ── Seed {seed} ──", flush=True)
        set_seed(seed)
        tstart = time.time()
        model, best_ep = train_model(
            data["X_train"], data["y_train_norm"],
            data["X_es"], data["y_es_norm"],
            n_feat, n_steps, device, f"s{seed}")
        # Per-step (not averaged) predictions on within and cross.
        p_w_norm = predict_all_steps(model, data["X_w"], device)
        p_c_norm = predict_all_steps(model, data["X_c"], device)
        # Denormalize per-step: broadcast (3,) over (N, 5, 3).
        p_w = p_w_norm * data["t_std"] + data["t_mean"]
        p_c = p_c_norm * data["t_std"] + data["t_mean"]

        path = save_predictions(seed, p_w, p_c,
                                data["y_w_raw"], data["y_c_raw"],
                                data["seq_w"], data["seq_c"])
        print(f"    Saved {path}  (best_ep={best_ep}, {time.time()-tstart:.0f}s)",
              flush=True)
        verify_avg_matches_baseline(seed)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  Total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
