#!/usr/bin/env python3
"""Category ablation on BEST_k80. Drops one feature category at a time,
reports within/cross r across 3 seeds."""
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


def load_best_k80():
    rk = pd.read_csv(RANK_TSV, sep="\t").sort_values("rank")
    return rk["feature"].tolist()[:K]


# ── Categorization ─────────────────────────────────────────────────
# Most specific first; first match wins.

def categorize(f):
    # Blob geometry
    if f.startswith("blob_"):
        return "Blob geometry"
    # Foreground spatial
    if f.startswith("fg_"):
        return "Foreground spatial"
    # Motion structure (specific motion + specific temporal_diff)
    if (f.startswith("motion_spatial") or f == "motion_concentration"
            or f.startswith("motion_vertical")
            or f in ("temporal_diff_skewness", "temporal_diff_kurtosis")):
        return "Motion structure"
    # Contour
    if f.startswith("contour_"):
        return "Contour"
    # Spectral
    if f.startswith("spectral_") or f.startswith("band_energy"):
        return "Spectral"
    # Visual clutter
    if f.startswith("subband_") or f.startswith("feature_congestion"):
        return "Visual clutter"
    # Foreground & motion (remainder)
    if ("foreground" in f or "shadow" in f
            or f.startswith("temporal_diff") or f.startswith("motion_pixel")):
        return "Foreground & motion"
    # MSCN & regional
    if (f.startswith("mscn") or f.startswith("top_") or f.startswith("mid_")
            or f.startswith("bot_") or f.startswith("ratio_top")):
        return "MSCN & regional"
    # GLCM
    if "glcm" in f:
        return "GLCM"
    # Edge & texture
    for kw in ("laplacian", "edge", "canny", "fast", "keypoint", "gradient", "rms"):
        if kw in f:
            return "Edge & texture"
    # Frequency
    for kw in ("fft", "dct", "gabor", "downsample", "spatial_frequency"):
        if kw in f:
            return "Frequency"
    # Color & illumination
    for kw in ("brightness", "dark_channel", "colorful", "saturation", "image_entropy"):
        if kw in f:
            return "Color & illumination"
    return None


# ── Data ───────────────────────────────────────────────────────────

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
    n_steps = H // SW; total = W + H; warmup = CFG["warmup_frames"]
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
        if epoch % 30 == 0 or pat >= CFG["patience"]:
            print(f"      [{label}] ep{epoch:3d} val={vloss:.6f} best={best:.6f} pat={pat}",
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


def run_config(df, feat_cols, label, device):
    print(f"\n  ── Config: {label}  (n_feat={len(feat_cols)}) ──", flush=True)
    tgt_cols = TARGETS
    n_steps = CFG["horizon"] // CFG["sub_window"]

    train_dfs, es_dfs, within_dfs = [], [], []
    for intx in CFG["train_intersections"]:
        tr, es, wv = split_intersection(df, intx)
        train_dfs.append(tr); es_dfs.append(es); within_dfs.append(wv)
    train_df = pd.concat(train_dfs, ignore_index=True)
    es_df = pd.concat(es_dfs, ignore_index=True)
    within_df = pd.concat(within_dfs, ignore_index=True)
    cross_df = df[df["sequence"] == CFG["test_intersection"]].copy()

    scaler = fit_scaler(train_df, feat_cols)
    X_train, y_train_raw = build_windows(train_df, feat_cols, tgt_cols,
                                          CFG["train_stride"], scaler)
    X_es, y_es_raw = build_windows(es_df, feat_cols, tgt_cols,
                                    CFG["eval_stride"], scaler)
    X_w, y_w_raw = build_windows(within_df, feat_cols, tgt_cols,
                                  CFG["eval_stride"], scaler)
    X_c, y_c_raw = build_windows(cross_df, feat_cols, tgt_cols,
                                  CFG["eval_stride"], scaler)

    y_flat = y_train_raw.reshape(-1, N_TARGETS)
    t_mean = y_flat.mean(axis=0)
    t_std = y_flat.std(axis=0); t_std[t_std < 1e-8] = 1.0
    y_train_norm = (y_train_raw - t_mean) / t_std
    y_es_norm = (y_es_raw - t_mean) / t_std

    y_w_mean = y_w_raw.mean(axis=1)
    y_c_mean = y_c_raw.mean(axis=1)

    n_feat = len(feat_cols)
    rows = []
    for seed in SEEDS:
        set_seed(seed)
        t0 = time.time()
        model, best_ep = train_model(
            X_train, y_train_norm, X_es, y_es_norm,
            n_feat, n_steps, device, f"{label}_s{seed}")
        pw = predict(model, X_w, device) * t_std + t_mean
        pc = predict(model, X_c, device) * t_std + t_mean
        elapsed = time.time() - t0
        row = {"config": label, "n_features": n_feat, "seed": seed,
                "best_epoch": best_ep, "elapsed_s": round(elapsed, 1)}
        for i, name in enumerate(tgt_cols):
            rw = float(np.corrcoef(pw[:, i], y_w_mean[:, i])[0, 1])
            rc = float(np.corrcoef(pc[:, i], y_c_mean[:, i])[0, 1])
            short = {"miss_rate_nano": "nano",
                      "recovery_nano_to_small": "ns",
                      "recovery_small_to_medium": "sm"}[name]
            row[f"within_r_{short}"] = round(rw, 6)
            row[f"cross_r_{short}"] = round(rc, 6)
        rows.append(row)
        print(f"    seed {seed}: within nano={row['within_r_nano']:.4f} "
              f"ns={row['within_r_ns']:.4f} sm={row['within_r_sm']:.4f}  "
              f"cross nano={row['cross_r_nano']:.4f} "
              f"ns={row['cross_r_ns']:.4f} sm={row['cross_r_sm']:.4f}  "
              f"(ep {best_ep}, {elapsed:.0f}s)", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    feats = load_best_k80()
    assert len(feats) == 80
    print(f"\n  BEST_k80 feature list ({len(feats)}):", flush=True)
    for i, f in enumerate(feats):
        print(f"    {i+1:2d}. {f}", flush=True)

    cat_map = {}
    uncategorized = []
    for f in feats:
        c = categorize(f)
        if c is None:
            uncategorized.append(f)
        else:
            cat_map.setdefault(c, []).append(f)

    if uncategorized:
        print(f"\n  ERROR: uncategorized features: {uncategorized}", flush=True)
        sys.exit(1)

    print(f"\n  Category assignment:", flush=True)
    total = 0
    for cat, fs in sorted(cat_map.items(), key=lambda x: -len(x[1])):
        print(f"    [{cat}] ({len(fs)}):", flush=True)
        for f in fs:
            print(f"        - {f}", flush=True)
        total += len(fs)
    assert total == 80, f"total={total}"
    print(f"\n  Total: {total} (ok)\n", flush=True)

    df = load_data()
    all_rows = []

    # Config 0: full k80
    all_rows.extend(run_config(df, feats, "full_k80", device))

    # Configs 1..N: drop each category
    for cat in cat_map.keys():
        drop = set(cat_map[cat])
        remaining = [f for f in feats if f not in drop]
        label = f"drop_{cat.replace(' & ', '_and_').replace(' ', '_')}"
        all_rows.extend(run_config(df, remaining, label, device))

        # Periodic save to guard against crashes
        tmp = pd.DataFrame(all_rows)
        tmp.to_csv(SCRIPT_DIR / "category_ablation.tsv", sep="\t", index=False)

    out = pd.DataFrame(all_rows)
    tsv = SCRIPT_DIR / "category_ablation.tsv"
    out.to_csv(tsv, sep="\t", index=False)
    print(f"\n  Saved {tsv}", flush=True)

    # Summary
    def agg(cfg):
        sub = [r for r in all_rows if r["config"] == cfg]
        if not sub:
            return None
        return {
            f"{k}_mean": float(np.mean([r[k] for r in sub]))
            for k in ["within_r_nano", "within_r_ns", "within_r_sm",
                      "cross_r_nano", "cross_r_ns", "cross_r_sm"]
        } | {
            f"{k}_std": float(np.std([r[k] for r in sub]))
            for k in ["within_r_nano", "within_r_ns", "within_r_sm",
                      "cross_r_nano", "cross_r_ns", "cross_r_sm"]
        }

    ref = agg("full_k80")
    print("\n" + "=" * 110, flush=True)
    print("  CATEGORY ABLATION SUMMARY (sorted by Δwithin_r_ns, largest drop first)",
          flush=True)
    print("=" * 110, flush=True)

    rows_sum = []
    for cat in cat_map.keys():
        label = f"drop_{cat.replace(' & ', '_and_').replace(' ', '_')}"
        a = agg(label)
        if a is None:
            continue
        rows_sum.append({
            "category": cat,
            "n_dropped": len(cat_map[cat]),
            "label": label,
            "within_r_ns_mean": a["within_r_ns_mean"],
            "within_r_ns_std": a["within_r_ns_std"],
            "within_r_nano_mean": a["within_r_nano_mean"],
            "within_r_sm_mean": a["within_r_sm_mean"],
            "cross_r_nano_mean": a["cross_r_nano_mean"],
            "cross_r_ns_mean": a["cross_r_ns_mean"],
            "cross_r_sm_mean": a["cross_r_sm_mean"],
            "d_within_r_ns": a["within_r_ns_mean"] - ref["within_r_ns_mean"],
            "d_cross_r_nano": a["cross_r_nano_mean"] - ref["cross_r_nano_mean"],
        })
    rows_sum.sort(key=lambda r: r["d_within_r_ns"])

    hdr = (f"  {'Category dropped':26s} | {'n':>2s} | "
           f"{'within_r_ns (mean±std)':23s} | {'Δr_ns':>7s} | "
           f"{'cross_r_nano':>12s} | {'Δcross':>7s}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    print(f"  {'(full k80)':26s} | {80:2d} | "
          f"{ref['within_r_ns_mean']:.4f}±{ref['within_r_ns_std']:.4f}        | "
          f"{'—':>7s} | {ref['cross_r_nano_mean']:12.4f} | {'—':>7s}",
          flush=True)
    for r in rows_sum:
        print(f"  {r['category']:26s} | {r['n_dropped']:2d} | "
              f"{r['within_r_ns_mean']:.4f}±{r['within_r_ns_std']:.4f}        | "
              f"{r['d_within_r_ns']:+7.4f} | "
              f"{r['cross_r_nano_mean']:12.4f} | "
              f"{r['d_cross_r_nano']:+7.4f}", flush=True)

    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
