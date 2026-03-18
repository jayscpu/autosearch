# Shared utility functions for the autosearch pipeline.
# Import these instead of duplicating across training/evaluation scripts.

import numpy as np
import pandas as pd


def load_data(features_csv, dets_csv, target="miss_rate"):
    """Load features CSV, merge detections, and compute the requested target.

    Parameters
    ----------
    features_csv : str or Path
        Path to the features CSV.
    dets_csv : str or Path
        Path to the detections CSV.
    target : str
        One of "fn_nano", "miss_rate", or "frame_f1".

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(features_csv)

    if target in ("miss_rate", "frame_f1"):
        dets = pd.read_csv(dets_csv)
        df = df.merge(dets[["frame_id", "nano_tp", "nano_fp", "x_count"]],
                      on="frame_id", how="left", suffixes=("", "_det"))
        for col in ["nano_tp", "nano_fp", "x_count"]:
            det_col = f"{col}_det"
            if det_col in df.columns:
                df[col] = df[det_col].fillna(df[col])
                df.drop(columns=[det_col], inplace=True)

        if target == "miss_rate":
            df["miss_rate"] = df["fn_nano"] / df["x_count"].clip(lower=1)
        elif target == "frame_f1":
            p = df["nano_tp"] / (df["nano_tp"] + df["nano_fp"]).clip(lower=1)
            r = df["nano_tp"] / (df["nano_tp"] + df["fn_nano"]).clip(lower=1)
            df["frame_f1"] = (2 * p * r) / (p + r).clip(lower=1e-8)
            df["frame_f1_inv"] = 1.0 - df["frame_f1"]

    return df


def transition_accuracy(y_true, y_pred):
    """Accuracy on frames where the true label changes."""
    trans_correct, trans_total = 0, 0
    for j in range(1, len(y_true)):
        if y_true[j] != y_true[j - 1]:
            trans_total += 1
            if y_pred[j] == y_true[j]:
                trans_correct += 1
    if trans_total == 0:
        return 0.0, 0
    return trans_correct / trans_total, trans_total
