#!/usr/bin/env python3
"""Merge 30 new spatial features into pod_features_all.csv -> pod_features_merged_95.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "second_autosearch" / "src"))
from pod_features import SPATIAL_65  # noqa: E402

ORIG_CSV = PROJECT / "second_autosearch" / "data" / "pod_features_all.csv"
OUT_CSV = ROOT / "pod_features_merged_95.csv"

INTERSECTION_FILES = [
    "features_30_Bellevue_116th_NE12th.csv",
    "features_30_Bellevue_150th_Eastgate.csv",
    "features_30_Bellevue_150th_Newport.csv",
    "features_30_Bellevue_150th_SE38th.csv",
    "features_30_Bellevue_Bellevue_NE8th.csv",
]

TEST_INTERSECTION = "Bellevue_150th_SE38th"

NEW_FEATURES_30 = [
    "blob_small_fraction", "blob_large_fraction", "blob_size_cv",
    "blob_compactness_mean", "blob_compactness_std",
    "blob_aspect_ratio_mean", "blob_aspect_ratio_std",
    "blob_nn_dist_mean", "blob_nn_dist_std", "blob_overlap_count",
    "fg_spatial_entropy", "fg_center_of_mass_y", "fg_center_of_mass_x",
    "fg_quadrant_imbalance", "fg_perimeter_ratio",
    "motion_spatial_entropy", "motion_concentration", "motion_vertical_bias",
    "temporal_diff_skewness", "temporal_diff_kurtosis",
    "contour_count", "contour_small_fraction", "contour_large_fraction",
    "contour_closed_ratio", "contour_hierarchy_depth",
    "spectral_centroid", "spectral_rolloff_85",
    "band_energy_fine_over_coarse", "subband_entropy", "feature_congestion",
]


def main():
    parts = []
    for fname in INTERSECTION_FILES:
        p = ROOT / fname
        df = pd.read_csv(p)
        print(f"  {fname}: {len(df)} rows, {len(df.columns)} cols", flush=True)
        parts.append(df)
    new_df = pd.concat(parts, ignore_index=True)
    print(f"  Concatenated new features: {new_df.shape}", flush=True)

    orig = pd.read_csv(ORIG_CSV)
    print(f"  Original features: {orig.shape}", flush=True)

    merged = orig.merge(new_df, on=["sequence", "video", "frame_id"], how="inner")
    print(f"  Merged (inner): {merged.shape}", flush=True)

    # Apply fixes using training-only percentiles
    train_mask = merged["sequence"] != TEST_INTERSECTION

    # 1) Clip fg_spatial_entropy to >= 0
    merged["fg_spatial_entropy"] = merged["fg_spatial_entropy"].clip(lower=0.0)

    # 2) Clip kurtosis and skewness at 99th percentile (training only)
    for col in ["temporal_diff_kurtosis", "temporal_diff_skewness"]:
        p99 = np.nanpercentile(merged.loc[train_mask, col].values, 99)
        print(f"    {col} 99th pct (train): {p99:.4f}  (pre max={merged[col].max():.4f})",
              flush=True)
        merged[col] = merged[col].clip(upper=float(p99))

    # Verification
    nan_counts = merged[NEW_FEATURES_30].isna().sum().sum()
    print(f"\n  Verification:", flush=True)
    print(f"    Rows: {len(merged)} (orig={len(orig)})  match={len(merged)==len(orig)}",
          flush=True)
    print(f"    Columns: {len(merged.columns)}", flush=True)
    print(f"    NaN in 30 new cols: {nan_counts}", flush=True)
    # Expected: 65 spatial + 30 new + bookkeeping (sequence/video/frame_id + targets etc.)
    spatial_cols_present = [c for c in SPATIAL_65 if c in merged.columns]
    new_cols_present = [c for c in NEW_FEATURES_30 if c in merged.columns]
    print(f"    SPATIAL_65 present: {len(spatial_cols_present)}/65", flush=True)
    print(f"    NEW_30 present: {len(new_cols_present)}/30", flush=True)
    print(f"    Total spatial features: {len(spatial_cols_present)+len(new_cols_present)}",
          flush=True)

    assert len(merged) == len(orig), "Row count mismatch!"
    assert nan_counts == 0, "NaN in new features!"
    assert len(spatial_cols_present) == 65
    assert len(new_cols_present) == 30

    merged.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}", flush=True)

    SPATIAL_95 = SPATIAL_65 + NEW_FEATURES_30
    print(f"\nSPATIAL_95 = SPATIAL_65 + {NEW_FEATURES_30}")
    print(f"  len(SPATIAL_95) = {len(SPATIAL_95)}")


if __name__ == "__main__":
    main()
