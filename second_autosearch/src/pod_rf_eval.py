#!/usr/bin/env python3
"""Run RF at a given config for the SPATIAL_65 sweep comparison."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pod_train_evid import (
    CONFIG, load_data, split_data, build_windows, build_rf_features,
    eval_regression, get_target_column,
)

def main():
    t0 = time.time()
    df = load_data()
    feature_cols = CONFIG["features"]

    train_df, earlystop_df, within_val_df, cross_val_df = split_data(df)

    warmup = CONFIG.get("warmup_frames", 0)
    if warmup > 0:
        import pandas as pd
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

    X_train, y_train, _, _ = build_windows(
        train_df, feature_cols, CONFIG["train_stride"], scaler)
    X_within, y_within, bnd_within, seq_labels_within = build_windows(
        within_val_df, feature_cols, CONFIG["eval_stride"], scaler)
    X_cross, y_cross, bnd_cross, _ = build_windows(
        cross_val_df, feature_cols, CONFIG["eval_stride"], scaler)

    # Thresholds
    if CONFIG.get("t1_absolute") is not None:
        t1 = float(CONFIG["t1_absolute"])
    else:
        t1 = float(np.percentile(y_train.flatten(), CONFIG["t1_percentile"]))
    if CONFIG.get("t2_absolute") is not None:
        t2 = float(CONFIG["t2_absolute"])
    else:
        t2 = float(np.percentile(y_train.flatten(), CONFIG["t2_percentile"]))

    print(f"  Thresholds: t1={t1:.4f}, t2={t2:.4f}", file=sys.stderr)
    print(f"  Windows: train={len(y_train)}, within={len(y_within)}, cross={len(y_cross)}",
          file=sys.stderr)

    # Build RF summary features
    X_train_rf = build_rf_features(X_train)
    X_within_rf = build_rf_features(X_within)
    X_cross_rf = build_rf_features(X_cross)
    y_train_mean = y_train.mean(axis=1)
    y_within_mean = y_within.mean(axis=1)
    y_cross_mean = y_cross.mean(axis=1)

    print(f"  RF features: {X_train_rf.shape[1]} (3x {len(feature_cols)} raw)", file=sys.stderr)

    rf = RandomForestRegressor(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        min_samples_leaf=CONFIG["rf_min_samples_leaf"],
        max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train_mean)

    pred_within = rf.predict(X_within_rf)
    pred_cross = rf.predict(X_cross_rf)

    m_within = eval_regression(y_within_mean, pred_within, t1, t2, bnd_within)
    m_cross = eval_regression(y_cross_mean, pred_cross, t1, t2, bnd_cross)

    elapsed = time.time() - t0
    print(f"  RF within: mse={m_within['mse']:.4f} cls_acc={m_within['cls_acc']:.3f} "
          f"cls_trans={m_within['cls_trans_acc']:.3f}", file=sys.stderr)
    print(f"  RF cross:  mse={m_cross['mse']:.4f} cls_acc={m_cross['cls_acc']:.3f} "
          f"cls_trans={m_cross['cls_trans_acc']:.3f}", file=sys.stderr)
    print(f"  Time: {elapsed:.1f}s", file=sys.stderr)

    config_summary = {
        "mode": "rf",
        "model": "RF",
        "n_features": len(feature_cols),
        "window": CONFIG["window"],
        "horizon": CONFIG["horizon"],
        "sub_window": CONFIG["sub_window"],
        "n_steps": CONFIG["horizon"] // CONFIG["sub_window"],
        "rf_n_estimators": CONFIG["rf_n_estimators"],
        "rf_max_depth": CONFIG["rf_max_depth"],
        "rf_min_samples_leaf": CONFIG["rf_min_samples_leaf"],
        "t1_percentile": CONFIG["t1_percentile"],
        "t2_percentile": CONFIG["t2_percentile"],
        "t1": round(t1, 4),
        "t2": round(t2, 4),
    }
    if CONFIG.get("single_cam"):
        config_summary["single_cam"] = CONFIG["single_cam"].split("_")[-1]

    result_fields = [
        "RESULT",
        f"model=RF",
        f"mse_within={m_within['mse']:.6f}",
        f"mse_cross={m_cross['mse']:.6f}",
        f"mae_within={m_within['mae']:.6f}",
        f"mae_cross={m_cross['mae']:.6f}",
        f"cls_acc_within={m_within['cls_acc']:.6f}",
        f"cls_acc_cross={m_cross['cls_acc']:.6f}",
        f"cls_trans_within={m_within['cls_trans_acc']:.6f}",
        f"cls_trans_cross={m_cross['cls_trans_acc']:.6f}",
        f"majority_acc_within={m_within['majority_acc']:.6f}",
        f"majority_acc_cross={m_cross['majority_acc']:.6f}",
        f"n_within={len(y_within)}",
        f"n_cross={len(y_cross)}",
        f"config={json.dumps(config_summary)}",
    ]
    print("\t".join(result_fields))


if __name__ == "__main__":
    main()
