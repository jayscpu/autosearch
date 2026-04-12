#!/usr/bin/env python3
"""3-seed comparison of 3 target formulations using BEST_k80 features.

Formulations
  1. 3-tier absolute: miss_rate_{nano,small,medium} — unclipped, unnormalized
  2. Recovery unclipped: nano, (nano-small), (small-medium) — unnormalized
  3. Recovery clipped+normalized: nano, (nano-small), max(0, small-medium) —
     target-normalized from training data (matches final_validation config).

All three use identical architecture/training protocol (h128/L2, dropout
0.4, lr 1e-3, wd 1e-5, bs 256, patience 40, epochs 300, grad_clip 1.0,
w30/h30/sw6, warmup 400, train_stride 15, eval_stride 30, 50/10/40 split,
SE38th held out) and the BEST_k80 feature subset from perm_importance_95.tsv.
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
FEATURES_CSV = SCRIPT_DIR / "pod_features_merged_95.csv"
RANK_TSV = SCRIPT_DIR / "perm_importance_95.tsv"
K = 80

sys.path.insert(0, str(PROJECT_DIR / "second_autosearch" / "src"))
from pod_features import SPATIAL_65  # noqa: E402


def load_best_k80():
    rk = pd.read_csv(RANK_TSV, sep="\t")
    rk = rk.sort_values("rank")
    feats = rk["feature"].tolist()[:K]
    assert len(feats) == K
    return feats


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
SEEDS = [42, 43, 44]

FORMULATIONS = [
    {
        "name": "3tier_absolute",
        "target_cols": ["miss_rate_nano", "miss_rate_small", "miss_rate_medium"],
        "normalize": False,
    },
    {
        "name": "recovery_unclipped",
        "target_cols": ["miss_rate_nano", "recovery_ns_raw", "recovery_sm_raw"],
        "normalize": False,
    },
    {
        "name": "recovery_clipped_normed",
        "target_cols": ["miss_rate_nano", "recovery_ns_raw", "recovery_sm_clipped"],
        "normalize": True,
    },
]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["miss_rate_nano"] = df["fn_nano"] / df["x_count"].clip(lower=1)
    df["miss_rate_small"] = df["fn_small"] / df["x_count"].clip(lower=1)
    df["miss_rate_medium"] = df["fn_medium"] / df["x_count"].clip(lower=1)
    df["recovery_ns_raw"] = df["miss_rate_nano"] - df["miss_rate_small"]
    df["recovery_sm_raw"] = df["miss_rate_small"] - df["miss_rate_medium"]
    df["recovery_sm_clipped"] = df["recovery_sm_raw"].clip(lower=0)
    return df


def split_intersection(df, name):
    sub = df[df["sequence"] == name].sort_values("frame_id")
    n = len(sub)
    t = int(n * CFG["train_fraction"])
    e = int(n * CFG["earlystop_fraction"])
    return sub.iloc[:t], sub.iloc[t:e], sub.iloc[e:]


def build_windows(df, feat_cols, tgt_cols, stride, scaler):
    W, H, SW = CFG["window"], CFG["horizon"], CFG["sub_window"]
    n_steps = H // SW
    total = W + H
    warmup = CFG["warmup_frames"]
    X, Y = [], []
    group_cols = ["sequence", "video"] if "video" in df.columns else ["sequence"]
    for _, sdf in df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        if warmup > 0:
            sdf = sdf.iloc[warmup:].reset_index(drop=True)
        if len(sdf) < total:
            continue
        fv = scaler.transform(sdf[feat_cols].values)
        tv = sdf[tgt_cols].values
        for t in range(0, len(sdf) - total + 1, stride):
            X.append(fv[t:t + W])
            tg = np.zeros((n_steps, len(tgt_cols)), dtype=np.float32)
            for step in range(n_steps):
                s = t + W + step * SW
                tg[step] = tv[s:s + SW].mean(axis=0)
            Y.append(tg)
    if not X:
        return (np.zeros((0, W, len(feat_cols)), dtype=np.float32),
                np.zeros((0, n_steps, len(tgt_cols)), dtype=np.float32))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def fit_scaler(train_df, feat_cols):
    group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
    kept = []
    for _, sdf in train_df.groupby(group_cols):
        sdf = sdf.sort_values("frame_id")
        kept.append(sdf.iloc[CFG["warmup_frames"]:])
    s = StandardScaler()
    s.fit(pd.concat(kept, ignore_index=True)[feat_cols].values)
    return s


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


def train_model(X_train, y_train, X_val, y_val, n_feat, n_steps, n_targets,
                 device, label):
    model = MultiOutputLSTM(n_feat, n_steps, n_targets,
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


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_formulation(df, feat_cols, formulation, device):
    name = formulation["name"]
    tgt_cols = formulation["target_cols"]
    normalize = formulation["normalize"]
    n_steps = CFG["horizon"] // CFG["sub_window"]
    n_feat = len(feat_cols)
    n_targets = len(tgt_cols)

    train_dfs, es_dfs, within_dfs = [], [], []
    for intx in CFG["train_intersections"]:
        tr, es, wv = split_intersection(df, intx)
        train_dfs.append(tr); es_dfs.append(es); within_dfs.append(wv)
    train_df = pd.concat(train_dfs, ignore_index=True)
    es_df = pd.concat(es_dfs, ignore_index=True)
    within_df = pd.concat(within_dfs, ignore_index=True)

    scaler = fit_scaler(train_df, feat_cols)
    X_train, y_train_raw = build_windows(train_df, feat_cols, tgt_cols,
                                          CFG["train_stride"], scaler)
    X_es, y_es_raw = build_windows(es_df, feat_cols, tgt_cols,
                                    CFG["eval_stride"], scaler)
    X_w, y_w_raw = build_windows(within_df, feat_cols, tgt_cols,
                                  CFG["eval_stride"], scaler)

    if normalize:
        y_flat = y_train_raw.reshape(-1, n_targets)
        t_mean = y_flat.mean(axis=0)
        t_std = y_flat.std(axis=0); t_std[t_std < 1e-8] = 1.0
        y_train_fit = (y_train_raw - t_mean) / t_std
        y_es_fit = (y_es_raw - t_mean) / t_std
    else:
        t_mean = np.zeros(n_targets)
        t_std = np.ones(n_targets)
        y_train_fit = y_train_raw.astype(np.float32)
        y_es_fit = y_es_raw.astype(np.float32)

    y_w_mean = y_w_raw.mean(axis=1)

    print(f"  [{name}] targets={tgt_cols} normalize={normalize}", flush=True)
    print(f"  Windows train={len(y_train_raw)} es={len(y_es_raw)} within={len(y_w_raw)}",
          flush=True)

    rows = []
    for seed in SEEDS:
        print(f"\n  ── {name} seed={seed} ──", flush=True)
        set_seed(seed)
        t0 = time.time()
        model, best_ep = train_model(
            X_train, y_train_fit, X_es, y_es_fit,
            n_feat, n_steps, n_targets, device, f"{name}_s{seed}")
        pred_w = predict(model, X_w, device) * t_std + t_mean
        elapsed = time.time() - t0

        rs = [float(np.corrcoef(pred_w[:, i], y_w_mean[:, i])[0, 1])
              for i in range(n_targets)]
        row = {
            "formulation": name, "seed": seed,
            "target1_name": tgt_cols[0], "r_target1": round(rs[0], 6),
            "target2_name": tgt_cols[1], "r_target2": round(rs[1], 6),
            "target3_name": tgt_cols[2], "r_target3": round(rs[2], 6),
            "best_epoch": best_ep, "elapsed_s": round(elapsed, 1),
        }
        rows.append(row)
        print(f"    r: {tgt_cols[0]}={rs[0]:.4f}  "
              f"{tgt_cols[1]}={rs[1]:.4f}  {tgt_cols[2]}={rs[2]:.4f}  "
              f"(ep {best_ep}, {elapsed:.0f}s)", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    feats = load_best_k80()
    print(f"  BEST_k80 features: {len(feats)}", flush=True)
    print(f"  (from SPATIAL_65: {sum(1 for f in feats if f in SPATIAL_65)}; "
          f"new: {sum(1 for f in feats if f not in SPATIAL_65)})", flush=True)

    df = load_data()
    print(f"  Data: {len(df)} rows\n", flush=True)

    all_rows = []
    for form in FORMULATIONS:
        print("=" * 70)
        print(f"  Formulation: {form['name']}")
        print("=" * 70)
        all_rows.extend(run_formulation(df, feats, form, device))

    out = pd.DataFrame(all_rows)
    tsv = SCRIPT_DIR / "formulation_comparison.tsv"
    out.to_csv(tsv, sep="\t", index=False)
    print(f"\n  Saved {tsv}", flush=True)

    # Summary
    print("\n" + "=" * 86)
    print("  SUMMARY (mean ± std across 3 seeds)")
    print("=" * 86)
    hdr = f"  {'Formulation':27s} | {'target 1':22s} | {'target 2':22s} | {'target 3':22s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for form in FORMULATIONS:
        sub = [r for r in all_rows if r["formulation"] == form["name"]]
        r1 = [r["r_target1"] for r in sub]
        r2 = [r["r_target2"] for r in sub]
        r3 = [r["r_target3"] for r in sub]
        n1 = form["target_cols"][0]; n2 = form["target_cols"][1]; n3 = form["target_cols"][2]
        print(f"  {form['name']:27s} | "
              f"{n1[:10]:10s} {np.mean(r1):.3f}±{np.std(r1):.3f} | "
              f"{n2[:10]:10s} {np.mean(r2):.3f}±{np.std(r2):.3f} | "
              f"{n3[:10]:10s} {np.mean(r3):.3f}±{np.std(r3):.3f}")

    print(f"\n  Total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
