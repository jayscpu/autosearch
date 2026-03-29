#!/usr/bin/env python3
"""Generate per-window predictions for the 5 best models and save to CSV.

Columns: frame_idx, pred_miss_rate, true_miss_rate, epistemic_unc (evidential only), intersection
"""

import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pod_features import ALL_FEATURES, SPATIAL_65, TOP_35_SPEARMAN

# Import from pod_train_evid (the version with transition_loss_weight support)
from pod_train_evid import (
    CONFIG, load_data, split_data, build_windows, build_rf_features,
    PlainLSTM, EvidentialLSTM,
    train_plain, train_evidential,
    nig_loss, nig_loss_weighted,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


# ── The 5 model configs ──────────────────────────────────────────

MODELS = {
    "plainlstm_4cam": {
        "label": "PlainLSTM 4CAM",
        "mode": "lstm",
        "overrides": {
            "features": SPATIAL_65,
            "n_features": 65,
            "hidden_size": 256,
            "n_layers": 4,
            "dropout": 0.4,
            "lr": 0.001,
            "batch_size": 64,
            "window": 30,
            "horizon": 30,
            "sub_window": 6,
            "train_stride": 6,
            "warmup_frames": 400,
            "t1_percentile": 25,
            "t2_percentile": 90,
            "single_cam": None,
            "seeds": [42, 123, 456],
            "transition_loss_weight": 0,
            "use_delta_features": False,
        },
    },
    "plainlstm_1cam_ne8th": {
        "label": "PlainLSTM 1CAM NE8th",
        "mode": "lstm",
        "overrides": {
            "features": TOP_35_SPEARMAN,
            "n_features": 35,
            "hidden_size": 96,
            "n_layers": 4,
            "dropout": 0.3,
            "lr": 0.003,
            "batch_size": 64,
            "window": 30,
            "horizon": 30,
            "sub_window": 6,
            "train_stride": 6,
            "warmup_frames": 300,
            "t1_percentile": 10,
            "t2_percentile": 70,
            "single_cam": "Bellevue_Bellevue_NE8th",
            "seeds": [42, 123, 456],
            "transition_loss_weight": 0,
            "use_delta_features": False,
        },
    },
    "evidential_4cam_tw8": {
        "label": "Evidential 4CAM TW w=8",
        "mode": "evidential",
        "overrides": {
            "features": SPATIAL_65,
            "n_features": 65,
            "hidden_size": 128,
            "n_layers": 4,
            "dropout": 0.4,
            "lr": 0.003,
            "batch_size": 64,
            "window": 30,
            "horizon": 30,
            "sub_window": 6,
            "train_stride": 6,
            "warmup_frames": 300,
            "t1_percentile": 20,
            "t2_percentile": 90,
            "single_cam": None,
            "lambda1": 0.28,
            "transition_loss_weight": 8,
            "use_delta_features": False,
        },
    },
    "rf_4cam": {
        "label": "RF 4CAM",
        "mode": "rf",
        "overrides": {
            "features": SPATIAL_65,
            "n_features": 65,
            "window": 30,
            "horizon": 30,
            "sub_window": 6,
            "train_stride": 6,
            "warmup_frames": 400,
            "t1_percentile": 25,
            "t2_percentile": 90,
            "single_cam": None,
            "rf_n_estimators": 500,
            "rf_max_depth": 20,
            "rf_min_samples_leaf": 10,
            "transition_loss_weight": 0,
            "use_delta_features": False,
        },
    },
    "rf_1cam_ne8th": {
        "label": "RF 1CAM NE8th",
        "mode": "rf",
        "overrides": {
            "features": TOP_35_SPEARMAN,
            "n_features": 35,
            "window": 30,
            "horizon": 30,
            "sub_window": 6,
            "train_stride": 6,
            "warmup_frames": 300,
            "t1_percentile": 10,
            "t2_percentile": 70,
            "single_cam": "Bellevue_Bellevue_NE8th",
            "rf_n_estimators": 500,
            "rf_max_depth": 20,
            "rf_min_samples_leaf": 10,
            "transition_loss_weight": 0,
            "use_delta_features": False,
        },
    },
}


def apply_config(overrides):
    """Apply overrides to global CONFIG dict."""
    for k, v in overrides.items():
        CONFIG[k] = v


def run_model(name, spec):
    """Train model and return predictions DataFrame."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {spec['label']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    apply_config(spec["overrides"])
    CONFIG["mode"] = spec["mode"]

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()
    feature_cols = CONFIG["features"]

    H = CONFIG["horizon"]
    S = CONFIG["sub_window"]
    n_steps = H // S

    train_df, earlystop_df, within_val_df, cross_val_df = split_data(df)

    # Scaler from training data (warmup-skipped)
    warmup = CONFIG.get("warmup_frames", 0)
    if warmup > 0:
        group_cols = ["sequence", "video"] if "video" in train_df.columns else ["sequence"]
        kept = []
        for _, sdf in train_df.groupby(group_cols):
            sdf = sdf.sort_values("frame_id")
            kept.append(sdf.iloc[warmup:])
        scaler_df = pd.concat(kept, ignore_index=True)
    else:
        scaler_df = train_df
    scaler = StandardScaler()
    scaler.fit(scaler_df[feature_cols].values)

    X_train, y_train, _, _, sw_train = build_windows(
        train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_earlystop, y_earlystop, _, _, _ = build_windows(
        earlystop_df, feature_cols, CONFIG["eval_stride"], scaler)
    X_within, y_within, bnd_within, seq_within, _ = build_windows(
        within_val_df, feature_cols, CONFIG["eval_stride"], scaler)
    X_cross, y_cross, bnd_cross, seq_cross, _ = build_windows(
        cross_val_df, feature_cols, CONFIG["eval_stride"], scaler)

    n_feat = len(feature_cols)

    # Thresholds
    t1 = float(np.percentile(y_train.flatten(), CONFIG["t1_percentile"]))
    t2 = float(np.percentile(y_train.flatten(), CONFIG["t2_percentile"]))

    print(f"  Windows: train={len(y_train)}, earlystop={len(y_earlystop)}, "
          f"within={len(y_within)}, cross={len(y_cross)}", file=sys.stderr)
    print(f"  Thresholds: t1={t1:.4f}, t2={t2:.4f}", file=sys.stderr)

    # Combine within + cross for eval
    is_single_cam = CONFIG.get("single_cam") is not None
    if is_single_cam:
        # In single-cam mode, cross==within, so just use within
        X_eval = X_within
        y_eval = y_within
        seq_eval = seq_within
    else:
        X_eval = np.concatenate([X_within, X_cross], axis=0)
        y_eval = np.concatenate([y_within, y_cross], axis=0)
        seq_eval = seq_within + seq_cross

    y_eval_mean = y_eval.mean(axis=1)
    mode = spec["mode"]

    if mode == "lstm":
        # PlainLSTM 3-seed ensemble
        all_preds = []
        for seed in CONFIG["seeds"]:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = train_plain(X_train, y_train, X_earlystop, y_earlystop,
                                n_feat, n_steps, device, sw_train)
            with torch.no_grad():
                p = model(torch.from_numpy(X_eval).to(device)).cpu().numpy()
            all_preds.append(p)
        pred_mean = np.mean(all_preds, axis=0).mean(axis=1)

        rows = []
        for i in range(len(y_eval_mean)):
            rows.append({
                "frame_idx": i,
                "pred_miss_rate": round(float(pred_mean[i]), 6),
                "true_miss_rate": round(float(y_eval_mean[i]), 6),
                "intersection": seq_eval[i],
            })

    elif mode == "evidential":
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = train_evidential(X_train, y_train, X_earlystop, y_earlystop,
                                 n_feat, n_steps, device, sw_train)

        val_x = torch.from_numpy(X_eval).to(device)
        with torch.no_grad():
            gamma, aleatoric, epistemic, nu, alpha, beta = model.predict(val_x)
        gamma_mean = gamma.cpu().numpy().mean(axis=1)
        epistemic_mean = epistemic.cpu().numpy().mean(axis=1)

        rows = []
        for i in range(len(y_eval_mean)):
            rows.append({
                "frame_idx": i,
                "pred_miss_rate": round(float(gamma_mean[i]), 6),
                "true_miss_rate": round(float(y_eval_mean[i]), 6),
                "epistemic_unc": round(float(epistemic_mean[i]), 6),
                "intersection": seq_eval[i],
            })

    elif mode == "rf":
        X_train_rf = build_rf_features(X_train)
        X_eval_rf = build_rf_features(X_eval)
        y_train_mean = y_train.mean(axis=1)

        rf = RandomForestRegressor(
            n_estimators=CONFIG["rf_n_estimators"],
            max_depth=CONFIG["rf_max_depth"],
            min_samples_leaf=CONFIG["rf_min_samples_leaf"],
            max_features="sqrt", n_jobs=-1, random_state=42)
        rf.fit(X_train_rf, y_train_mean)
        pred = rf.predict(X_eval_rf)

        rows = []
        for i in range(len(y_eval_mean)):
            rows.append({
                "frame_idx": i,
                "pred_miss_rate": round(float(pred[i]), 6),
                "true_miss_rate": round(float(y_eval_mean[i]), 6),
                "intersection": seq_eval[i],
            })

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, {len(rows)} predictions", file=sys.stderr)

    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, spec in MODELS.items():
        pred_df = run_model(name, spec)
        outpath = RESULTS_DIR / f"predictions_{name}.csv"
        pred_df.to_csv(outpath, index=False)
        print(f"  Saved {outpath} ({len(pred_df)} rows)", file=sys.stderr)

    print("\nAll predictions saved.", file=sys.stderr)


if __name__ == "__main__":
    main()
