#!/usr/bin/env python3
"""
Compare three decoder architectures for clip_sm_normed (h128/L2):
  Baseline:  shared decoder + shared head (current architecture)
  Variant A: shared decoder + separate heads per target
  Variant B: separate decoders + separate heads per target

All use same encoder, same config, same data pipeline.
3-seed validation (42, 43, 44). Within/cross reported separately.
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
SEEDS = [42, 43, 44]
TSV_PATH = SCRIPT_DIR / "results.tsv"
TSV_COLUMNS = [
    "variant", "seed",
    "within_r_nano", "within_r_ns", "within_r_sm",
    "cross_r_nano", "cross_r_ns", "cross_r_sm",
    "best_epoch", "elapsed_s",
]


def _fsync_write(path, mode, content):
    with open(path, mode) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


# ═══════════════════════════════════════════════════════════════════
# DATA (copied from final_validation/run_experiments.py)
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
# MODELS
# ═══════════════════════════════════════════════════════════════════

class BaselineModel(nn.Module):
    """Current architecture: shared decoder + shared head."""
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


class VariantA(nn.Module):
    """Shared decoder + separate heads per target."""
    def __init__(self, input_size, n_steps, n_targets, hidden, n_layers, dropout):
        super().__init__()
        self.n_steps, self.n_targets = n_steps, n_targets
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.decoder = nn.LSTMCell(input_size=n_targets, hidden_size=hidden)
        self.head_nano = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_ns = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_sm = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_dec, c_dec = h_n[-1], c_n[-1]
        preds = []
        dec_input = torch.zeros(x.size(0), self.n_targets, device=x.device)
        for _ in range(self.n_steps):
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            p_nano = self.head_nano(h_dec)
            p_ns = self.head_ns(h_dec)
            p_sm = self.head_sm(h_dec)
            step_pred = torch.cat([p_nano, p_ns, p_sm], dim=1)
            preds.append(step_pred)
            dec_input = step_pred.detach()
        return torch.stack(preds, dim=1)


class VariantB(nn.Module):
    """Separate decoders + separate heads per target."""
    def __init__(self, input_size, n_steps, n_targets, hidden, n_layers, dropout):
        super().__init__()
        self.n_steps, self.n_targets = n_steps, n_targets
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden,
                               num_layers=n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.decoder_nano = nn.LSTMCell(input_size=1, hidden_size=hidden)
        self.decoder_ns = nn.LSTMCell(input_size=1, hidden_size=hidden)
        self.decoder_sm = nn.LSTMCell(input_size=1, hidden_size=hidden)
        self.head_nano = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_ns = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_sm = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_enc, c_enc = h_n[-1], c_n[-1]
        # Each decoder starts from same encoder state
        h_nano, c_nano = h_enc.clone(), c_enc.clone()
        h_ns, c_ns = h_enc.clone(), c_enc.clone()
        h_sm, c_sm = h_enc.clone(), c_enc.clone()
        inp_nano = torch.zeros(x.size(0), 1, device=x.device)
        inp_ns = torch.zeros(x.size(0), 1, device=x.device)
        inp_sm = torch.zeros(x.size(0), 1, device=x.device)
        preds = []
        for _ in range(self.n_steps):
            h_nano, c_nano = self.decoder_nano(inp_nano, (h_nano, c_nano))
            h_ns, c_ns = self.decoder_ns(inp_ns, (h_ns, c_ns))
            h_sm, c_sm = self.decoder_sm(inp_sm, (h_sm, c_sm))
            p_nano = self.head_nano(h_nano)
            p_ns = self.head_ns(h_ns)
            p_sm = self.head_sm(h_sm)
            inp_nano = p_nano.detach()
            inp_ns = p_ns.detach()
            inp_sm = p_sm.detach()
            preds.append(torch.cat([p_nano, p_ns, p_sm], dim=1))
        return torch.stack(preds, dim=1)


MODEL_CLASSES = {
    "baseline": BaselineModel,
    "variant_A": VariantA,
    "variant_B": VariantB,
}


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(model_cls, X_train, y_train, X_val, y_val, n_feat, n_steps,
                device, label):
    hidden = CONFIG["hidden_size"]
    n_layers = CONFIG["n_layers"]
    dropout = CONFIG["dropout"]
    model = model_cls(n_feat, n_steps, N_TARGETS, hidden, n_layers, dropout).to(device)
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


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    df = load_data()
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
    X_within, y_within_raw, _ = build_windows(within_df, feat_cols, tgt_cols,
                                               W, H, SW, CONFIG["eval_stride"],
                                               scaler, warmup)
    X_cross, y_cross_raw, _ = build_windows(cross_df, feat_cols, tgt_cols,
                                             W, H, SW, CONFIG["eval_stride"],
                                             scaler, warmup)

    y_within_mean = y_within_raw.mean(axis=1)
    y_cross_mean = y_cross_raw.mean(axis=1)

    # Target normalization from training
    y_flat = y_train_raw.reshape(-1, N_TARGETS)
    t_mean = y_flat.mean(axis=0)
    t_std = y_flat.std(axis=0)
    t_std[t_std < 1e-8] = 1.0
    y_train_norm = (y_train_raw - t_mean) / t_std
    y_es_norm = (y_es_raw - t_mean) / t_std

    n_feat = len(feat_cols)
    print(f"  Windows: train={len(y_train_raw)}, es={len(y_es_raw)}, "
          f"within={len(y_within_raw)}, cross={len(y_cross_raw)}", flush=True)

    # Resume support
    done_ids = set()
    if TSV_PATH.exists():
        try:
            existing = pd.read_csv(TSV_PATH, sep="\t")
            done_ids = {f"{r['variant']}_s{r['seed']}" for _, r in existing.iterrows()}
            print(f"  Resuming: {len(done_ids)} runs already done", flush=True)
        except Exception:
            pass
    else:
        _fsync_write(TSV_PATH, "w", "\t".join(TSV_COLUMNS) + "\n")

    all_results = []

    for variant_name, model_cls in MODEL_CLASSES.items():
        for seed in SEEDS:
            run_id = f"{variant_name}_s{seed}"
            if run_id in done_ids:
                print(f"  Skipping {run_id} (already done)", flush=True)
                continue

            print(f"\n  === {run_id} ===", flush=True)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            t0 = time.time()
            model, best_ep = train_model(model_cls, X_train, y_train_norm,
                                          X_es, y_es_norm, n_feat, n_steps,
                                          device, run_id)

            pred_within_norm = predict(model, X_within, device)
            pred_cross_norm = predict(model, X_cross, device)
            pred_within = pred_within_norm * t_std + t_mean
            pred_cross = pred_cross_norm * t_std + t_mean

            elapsed = time.time() - t0

            # Correlations
            wr = [float(np.corrcoef(pred_within[:, i], y_within_mean[:, i])[0, 1])
                  for i in range(N_TARGETS)]
            cr = [float(np.corrcoef(pred_cross[:, i], y_cross_mean[:, i])[0, 1])
                  for i in range(N_TARGETS)]

            print(f"    Within: nano={wr[0]:.4f}  ns={wr[1]:.4f}  sm={wr[2]:.4f}", flush=True)
            print(f"    Cross:  nano={cr[0]:.4f}  ns={cr[1]:.4f}  sm={cr[2]:.4f}", flush=True)
            print(f"    ep{best_ep} ({elapsed:.0f}s)", flush=True)

            row = {
                "variant": variant_name, "seed": seed,
                "within_r_nano": round(wr[0], 6),
                "within_r_ns": round(wr[1], 6),
                "within_r_sm": round(wr[2], 6),
                "cross_r_nano": round(cr[0], 6),
                "cross_r_ns": round(cr[1], 6),
                "cross_r_sm": round(cr[2], 6),
                "best_epoch": best_ep,
                "elapsed_s": round(elapsed, 1),
            }
            _fsync_write(TSV_PATH, "a",
                         "\t".join(str(row[c]) for c in TSV_COLUMNS) + "\n")
            all_results.append(row)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Summary table ──
    all_df = pd.read_csv(TSV_PATH, sep="\t")

    print(f"\n{'='*90}")
    print(f"  SUMMARY: mean ± std across 3 seeds")
    print(f"{'='*90}")
    print(f"  {'Variant':12s} | {'W nano':>12s} {'W ns':>12s} {'W sm':>12s} | "
          f"{'C nano':>12s} {'C ns':>12s} {'C sm':>12s}")
    print(f"  {'-'*86}")

    for variant in ["baseline", "variant_A", "variant_B"]:
        vdf = all_df[all_df["variant"] == variant]
        if len(vdf) == 0:
            continue
        parts = []
        for col in ["within_r_nano", "within_r_ns", "within_r_sm",
                     "cross_r_nano", "cross_r_ns", "cross_r_sm"]:
            m, s = vdf[col].mean(), vdf[col].std()
            parts.append(f"{m:.3f}±{s:.3f}")
        print(f"  {variant:12s} | {parts[0]:>12s} {parts[1]:>12s} {parts[2]:>12s} | "
              f"{parts[3]:>12s} {parts[4]:>12s} {parts[5]:>12s}")

    elapsed_total = time.time() - t_start
    print(f"\n  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
