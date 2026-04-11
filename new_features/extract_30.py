#!/usr/bin/env python3
"""Extract 30 new features from all intersection videos.

Features are extracted per-frame using MOG2 background subtraction,
connected components, Canny edges, FFT, Gabor filters, etc.
Parallelized across videos using multiprocessing.Pool(16).
"""

import os
import sys
import time
import glob
import subprocess
import threading
import datetime
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
DATA_DIR = Path("/workspace/autosearch/pod/data")
REPO_DIR = Path("/home/runner/autosearch_repo")
DET_DIR = REPO_DIR / "second_autosearch" / "data"
OUTPUT_DIR = REPO_DIR / "new_features"
FRAMES_DIR = REPO_DIR / "new_features" / "frames"

INTERSECTIONS = [
    "Bellevue_116th_NE12th",
    "Bellevue_150th_Eastgate",
    "Bellevue_150th_Newport",
    "Bellevue_150th_SE38th",
    "Bellevue_Bellevue_NE8th",
]

TOTAL_PX = 1280 * 720

FEATURE_COLS = [
    "blob_small_fraction", "blob_large_fraction", "blob_size_cv",
    "blob_compactness_mean", "blob_compactness_std",
    "blob_aspect_ratio_mean", "blob_aspect_ratio_std",
    "blob_nn_dist_mean", "blob_nn_dist_std", "blob_overlap_count",
    "fg_spatial_entropy", "fg_center_of_mass_y", "fg_center_of_mass_x",
    "fg_quadrant_imbalance", "fg_perimeter_ratio",
    "motion_spatial_entropy", "motion_concentration",
    "motion_vertical_bias", "temporal_diff_skewness", "temporal_diff_kurtosis",
    "contour_count", "contour_small_fraction", "contour_large_fraction",
    "contour_closed_ratio", "contour_hierarchy_depth",
    "spectral_centroid", "spectral_rolloff_85", "band_energy_fine_over_coarse",
    "subband_entropy", "feature_congestion",
]

# ═══════════════════════════════════════════════════════════
# PRECOMPUTE GABOR KERNELS
# ═══════════════════════════════════════════════════════════
def build_gabor_kernels():
    kernels = []
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for theta in angles:
        k1 = cv2.getGaborKernel((21,21), 3.0, theta, 10, 0.5)
        k2 = cv2.getGaborKernel((11,11), 1.5, theta, 5, 0.5)
        kernels.append(k1)
        kernels.append(k2)
    return kernels

GABOR_KERNELS = build_gabor_kernels()

# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION (per-frame)
# ═══════════════════════════════════════════════════════════

def extract_30_features(bgr, gray, fg_clean, n_labels, labels, stats, centroids,
                        edges, temporal_diff, has_prev, gabor_kernels):
    feats = {}

    # --- GROUP A: Blob Geometry ---
    if n_labels > 1:
        areas_all = stats[1:, cv2.CC_STAT_AREA]
        valid_mask = (areas_all >= 200) & (areas_all < 0.3 * TOTAL_PX)
        n_valid = int(valid_mask.sum())
    else:
        n_valid = 0

    if n_valid == 0:
        for k in FEATURE_COLS[:10]:
            feats[k] = 0.0
    else:
        valid_areas = areas_all[valid_mask]
        valid_labels = np.where(valid_mask)[0] + 1  # +1 because we skipped label 0
        valid_cents = centroids[1:][valid_mask]  # centroids includes label 0

        # 1-2: size fractions
        feats["blob_small_fraction"] = float(np.sum(valid_areas < 500) / n_valid)
        feats["blob_large_fraction"] = float(np.sum(valid_areas > 5000) / n_valid)

        # 3: size CV
        feats["blob_size_cv"] = float(np.std(valid_areas) / (np.mean(valid_areas) + 1e-8))

        # 4-5: compactness (subsample if > 200)
        if n_valid > 200:
            idx_sub = np.random.choice(n_valid, 200, replace=False)
        else:
            idx_sub = np.arange(n_valid)

        compactness_vals = []
        for ii in idx_sub:
            lab = valid_labels[ii]
            blob_mask = (labels == lab).astype(np.uint8)
            contours_i, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_i:
                peri = cv2.arcLength(contours_i[0], True)
            else:
                peri = 0
            comp = 4 * np.pi * valid_areas[ii] / (peri**2 + 1e-8)
            compactness_vals.append(comp)
        compactness_vals = np.array(compactness_vals)
        feats["blob_compactness_mean"] = float(compactness_vals.mean())
        feats["blob_compactness_std"] = float(compactness_vals.std())

        # 6-7: aspect ratio
        ar_vals = stats[valid_labels, cv2.CC_STAT_WIDTH].astype(float) / (
            stats[valid_labels, cv2.CC_STAT_HEIGHT].astype(float) + 1e-8)
        feats["blob_aspect_ratio_mean"] = float(ar_vals.mean())
        feats["blob_aspect_ratio_std"] = float(ar_vals.std())

        # 8-9: nearest-neighbor distance
        if n_valid <= 1:
            feats["blob_nn_dist_mean"] = 0.0
            feats["blob_nn_dist_std"] = 0.0
        else:
            dists = cdist(valid_cents, valid_cents)
            np.fill_diagonal(dists, np.inf)
            nn_dists = dists.min(axis=1)
            feats["blob_nn_dist_mean"] = float(nn_dists.mean())
            feats["blob_nn_dist_std"] = float(nn_dists.std())

        # 10: overlap count
        if n_valid <= 1:
            feats["blob_overlap_count"] = 0.0
        else:
            xs = stats[valid_labels, cv2.CC_STAT_LEFT]
            ys = stats[valid_labels, cv2.CC_STAT_TOP]
            ws = stats[valid_labels, cv2.CC_STAT_WIDTH]
            hs = stats[valid_labels, cv2.CC_STAT_HEIGHT]
            overlap = 0
            for i in range(n_valid):
                for j in range(i+1, n_valid):
                    if (xs[i] < xs[j]+ws[j] and xs[j] < xs[i]+ws[i] and
                        ys[i] < ys[j]+hs[j] and ys[j] < ys[i]+hs[i]):
                        overlap += 1
            feats["blob_overlap_count"] = float(overlap)

    # --- GROUP B: Foreground Spatial Distribution ---
    fg_sum = fg_clean.sum()
    if fg_sum == 0:
        feats["fg_spatial_entropy"] = 0.0
        feats["fg_center_of_mass_y"] = 0.5
        feats["fg_center_of_mass_x"] = 0.5
        feats["fg_quadrant_imbalance"] = 0.0
        feats["fg_perimeter_ratio"] = 0.0
    else:
        # 11: spatial entropy (4x4 grid)
        fractions = np.zeros(16)
        cell_h, cell_w = 180, 320
        for r in range(4):
            for c in range(4):
                cell = fg_clean[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                fractions[r*4+c] = cell.sum() / (cell_h * cell_w)
        total_frac = fractions.sum()
        if total_frac < 1e-10:
            feats["fg_spatial_entropy"] = 0.0
        else:
            p = fractions / total_frac
            feats["fg_spatial_entropy"] = float(-np.sum(p * np.log2(p + 1e-10)))

        # 12-13: center of mass
        ys, xs = np.where(fg_clean > 0)
        feats["fg_center_of_mass_y"] = float(ys.mean() / 720)
        feats["fg_center_of_mass_x"] = float(xs.mean() / 1280)

        # 14: quadrant imbalance
        h2, w2 = 360, 640
        q = np.array([
            fg_clean[:h2, :w2].sum(),
            fg_clean[:h2, w2:].sum(),
            fg_clean[h2:, :w2].sum(),
            fg_clean[h2:, w2:].sum(),
        ], dtype=float) / (h2 * w2)
        feats["fg_quadrant_imbalance"] = float(q.max() - q.min())

        # 15: perimeter ratio
        dilated = cv2.dilate(fg_clean, np.ones((3,3), np.uint8))
        boundary = cv2.bitwise_xor(dilated, fg_clean)
        feats["fg_perimeter_ratio"] = float(boundary.sum() / TOTAL_PX)

    # --- GROUP C: Motion Structure ---
    if not has_prev:
        feats["motion_spatial_entropy"] = 0.0
        feats["motion_concentration"] = 1.0
        feats["motion_vertical_bias"] = 0.0
        feats["temporal_diff_skewness"] = 0.0
        feats["temporal_diff_kurtosis"] = 0.0
    else:
        td = temporal_diff
        total_motion = td.sum()

        if total_motion < 1.0:
            feats["motion_spatial_entropy"] = 0.0
            feats["motion_concentration"] = 1.0
            feats["motion_vertical_bias"] = 0.0
        else:
            # 16: motion spatial entropy
            cell_h, cell_w = 180, 320
            cell_sums = np.zeros(16)
            for r in range(4):
                for c in range(4):
                    cell_sums[r*4+c] = td[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w].sum()
            cs_total = cell_sums.sum()
            p = cell_sums / (cs_total + 1e-10)
            feats["motion_spatial_entropy"] = float(-np.sum(p * np.log2(p + 1e-10)))

            # 17: motion concentration
            sorted_cs = np.sort(cell_sums)[::-1]
            cumsum = np.cumsum(sorted_cs)
            threshold = 0.8 * cs_total
            k = int(np.searchsorted(cumsum, threshold) + 1)
            feats["motion_concentration"] = float(k / 16)

            # 18: vertical bias
            top_sum = td[:360, :].sum()
            bot_sum = td[360:, :].sum()
            feats["motion_vertical_bias"] = float((top_sum - bot_sum) / (total_motion + 1e-8))

        # 19-20: skewness and kurtosis (numpy for speed)
        flat = td.ravel()
        std_val = flat.std()
        if std_val < 1e-8:
            feats["temporal_diff_skewness"] = 0.0
            feats["temporal_diff_kurtosis"] = 0.0
        else:
            m = flat.mean()
            z = (flat - m) / std_val
            n = len(z)
            feats["temporal_diff_skewness"] = float(np.mean(z**3))
            feats["temporal_diff_kurtosis"] = float(np.mean(z**4) - 3.0)

    # --- GROUP D: Contour Geometry ---
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n_contours = len(contours)
    if n_contours == 0:
        feats["contour_count"] = 0.0
        feats["contour_small_fraction"] = 0.0
        feats["contour_large_fraction"] = 0.0
        feats["contour_closed_ratio"] = 0.0
        feats["contour_hierarchy_depth"] = 0.0
    else:
        feats["contour_count"] = float(n_contours)

        # Subsample contours if too many (> 500) for perimeter/closed computation
        if n_contours > 500:
            sample_idx = np.random.choice(n_contours, 500, replace=False)
            sample_contours = [contours[i] for i in sample_idx]
        else:
            sample_contours = contours
        n_sample = len(sample_contours)

        perimeters = np.array([cv2.arcLength(c, False) for c in sample_contours])
        feats["contour_small_fraction"] = float(np.sum(perimeters < 50) / n_sample)
        feats["contour_large_fraction"] = float(np.sum(perimeters > 200) / n_sample)

        # closed ratio
        closed = 0
        for c in sample_contours:
            if len(c) >= 2:
                dx = float(c[0][0][0]) - float(c[-1][0][0])
                dy = float(c[0][0][1]) - float(c[-1][0][1])
                if dx*dx + dy*dy < 25:  # < 5 pixels
                    closed += 1
        feats["contour_closed_ratio"] = float(closed / n_sample)

        # hierarchy depth — vectorized with parent array
        h_arr = hierarchy[0]  # shape (N, 4)
        parents = h_arr[:, 3]
        depths = np.zeros(n_contours, dtype=np.int32)
        for i in range(n_contours):
            d = 0
            idx = i
            while parents[idx] >= 0:
                d += 1
                idx = parents[idx]
                if d > 50:
                    break
            depths[i] = d
        feats["contour_hierarchy_depth"] = float(depths.max())

    # --- GROUP E: Scale-Specific Frequency (downsampled for speed) ---
    # Downsample to 320x180 — spectral ratios are scale-invariant
    gray_small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    gray_sf = gray_small.astype(np.float64)
    f = np.fft.fft2(gray_sf)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    sh, sw = gray_small.shape
    cy, cx = sh // 2, sw // 2
    Y, X = np.ogrid[:sh, :sw]
    radius = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_radius = np.sqrt(cx**2 + cy**2)

    total_mag = magnitude.sum()
    if total_mag < 1e-8:
        feats["spectral_centroid"] = 0.0
        feats["spectral_rolloff_85"] = 0.0
        feats["band_energy_fine_over_coarse"] = 0.0
    else:
        feats["spectral_centroid"] = float(np.sum(radius * magnitude) / (total_mag + 1e-8) / max_radius)

        radii_flat = radius.ravel()
        mag_flat = magnitude.ravel()
        order = np.argsort(radii_flat)
        cumsum_mag = np.cumsum(mag_flat[order])
        total = cumsum_mag[-1]
        idx_85 = np.searchsorted(cumsum_mag, 0.85 * total)
        idx_85 = min(idx_85, len(order) - 1)
        feats["spectral_rolloff_85"] = float(radii_flat[order[idx_85]] / max_radius)

        band_coarse = magnitude[radius <= 0.25 * max_radius].sum()
        band_fine = magnitude[radius >= 0.75 * max_radius].sum()
        feats["band_energy_fine_over_coarse"] = float(band_fine / (band_coarse + 1e-8))

    # --- GROUP F: Visual Clutter (downsampled for speed) ---
    # 29: subband entropy — Gabor on 320x180
    energy_maps = []
    for kernel in gabor_kernels:
        response = cv2.filter2D(gray_sf, cv2.CV_64F, kernel)
        energy = response ** 2
        smoothed = cv2.blur(energy, (4, 4))  # scaled down from (16,16)
        energy_maps.append(smoothed)
    stack = np.stack(energy_maps, axis=-1)  # (180, 320, 8)
    channel_sum = stack.sum(axis=-1, keepdims=True) + 1e-10
    p = stack / channel_sum
    # Clip to avoid log2 warnings on near-zero values
    np.clip(p, 1e-10, None, out=p)
    pixel_entropy = -np.sum(p * np.log2(p), axis=-1)
    feats["subband_entropy"] = float(pixel_entropy.mean())

    # 30: feature congestion — on 320x180
    bgr_small = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
    lum = gray_sf
    lum_sq = cv2.blur(lum**2, (4, 4))
    lum_mean = cv2.blur(lum, (4, 4))
    lum_var = np.maximum(lum_sq - lum_mean**2, 0)

    grad = np.sqrt(cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3)**2 +
                   cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3)**2)
    grad_sq = cv2.blur(grad**2, (4, 4))
    grad_mean = cv2.blur(grad, (4, 4))
    grad_var = np.maximum(grad_sq - grad_mean**2, 0)

    hsv_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    sat = hsv_small[:, :, 1].astype(np.float64)
    sat_sq = cv2.blur(sat**2, (4, 4))
    sat_mean = cv2.blur(sat, (4, 4))
    sat_var = np.maximum(sat_sq - sat_mean**2, 0)

    congestion = np.cbrt(lum_var * grad_var * sat_var)
    feats["feature_congestion"] = float(congestion.mean())

    return feats


# ═══════════════════════════════════════════════════════════
# VIDEO WORKER
# ═══════════════════════════════════════════════════════════

def process_one_video(args):
    intersection_name, video_path, frames_base_dir, expected_n_frames, gabor_kernels = args
    video_path = Path(video_path)
    vname = video_path.stem

    # Extract frames if needed
    vid_frames_dir = Path(frames_base_dir) / intersection_name / vname
    vid_frames_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(glob.glob(str(vid_frames_dir / "frame_*.jpg")))
    if len(existing_frames) < 1:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "fps=2,scale=1280:720", "-q:v", "2",
            str(vid_frames_dir / "frame_%06d.jpg")
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR ffmpeg {intersection_name}/{vname}: {result.stderr[-200:]}", flush=True)
            return []
        existing_frames = sorted(glob.glob(str(vid_frames_dir / "frame_*.jpg")))

    n_actual = len(existing_frames)
    if n_actual != expected_n_frames:
        print(f"  WARNING frame count mismatch {intersection_name}/{vname}: "
              f"expected={expected_n_frames} actual={n_actual}", flush=True)

    n_use = min(n_actual, expected_n_frames)
    frame_files = existing_frames[:n_use]

    # Process frames
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    prev_gray = None
    results = []

    t0 = time.time()
    for i, fpath in enumerate(frame_files):
        bgr = cv2.imread(fpath)
        if bgr is None:
            # append zeros row
            row = {"sequence": intersection_name, "video": vname, "frame_index": i}
            for k in FEATURE_COLS:
                row[k] = 0.0
            results.append(row)
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # MOG2
        fg_mask = mog.apply(bgr)
        fg_binary = (fg_mask == 255).astype(np.uint8)
        fg_clean = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, kernel)

        # Connected components
        n_labels, labels_map, stats, centroids_cc = cv2.connectedComponentsWithStats(
            fg_clean, connectivity=8)

        # Canny
        edges = cv2.Canny(gray, 50, 150)

        # Temporal diff
        if prev_gray is not None:
            temporal_diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
            has_prev = True
        else:
            temporal_diff = np.zeros_like(gray, dtype=np.float32)
            has_prev = False

        feats = extract_30_features(
            bgr, gray, fg_clean, n_labels, labels_map, stats, centroids_cc,
            edges, temporal_diff, has_prev, gabor_kernels)

        feats["sequence"] = intersection_name
        feats["video"] = vname
        feats["frame_index"] = i
        results.append(feats)

        prev_gray = gray

    elapsed = time.time() - t0
    fps = n_use / elapsed if elapsed > 0 else 0
    print(f"  {intersection_name}/{vname}: {n_use} frames in {elapsed:.1f}s ({fps:.1f} fps)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════
# GIT AUTOCOMMIT THREAD
# ═══════════════════════════════════════════════════════════
_stop_autocommit = threading.Event()

def autocommit_loop():
    while not _stop_autocommit.is_set():
        _stop_autocommit.wait(300)  # every 5 minutes
        if _stop_autocommit.is_set():
            break
        try:
            subprocess.run(["git", "add", "new_features/*.csv"],
                           cwd=str(REPO_DIR), check=False, capture_output=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            r = subprocess.run(
                ["git", "commit", "-m", f"extraction checkpoint {ts}"],
                cwd=str(REPO_DIR), check=False, capture_output=True, text=True)
            if r.returncode == 0:
                subprocess.run(["git", "push"], cwd=str(REPO_DIR), check=False, capture_output=True)
                print(f"  [AUTOCOMMIT] checkpoint at {ts}", flush=True)
        except Exception as e:
            print(f"  [AUTOCOMMIT] error: {e}", flush=True)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # 1. Load detection CSVs → build reference
    print("Loading detection CSVs...", flush=True)
    ref = {}  # (intersection, video) -> sorted list of frame_ids
    total_frames = 0
    for inter in INTERSECTIONS:
        det_path = DET_DIR / f"pod_detections_{inter}.csv"
        df = pd.read_csv(det_path)
        for video, grp in df.groupby("video"):
            fids = grp["frame_id"].tolist()
            ref[(inter, video)] = fids
            total_frames += len(fids)
    print(f"  Total reference frames: {total_frames}", flush=True)

    gabor_kernels = build_gabor_kernels()

    # 2. Start autocommit thread
    ac_thread = threading.Thread(target=autocommit_loop, daemon=True)
    ac_thread.start()

    # 3. Process intersection by intersection (disk space management)
    frames_done_total = 0
    completed_videos_total = 0
    total_videos = 0
    for inter in INTERSECTIONS:
        video_dir = DATA_DIR / inter
        videos = sorted(video_dir.glob("*.mp4"))
        for vp in videos:
            if (inter, vp.stem) in ref:
                total_videos += 1

    for inter in INTERSECTIONS:
        out_csv = OUTPUT_DIR / f"features_30_{inter}.csv"
        expected_rows = sum(len(v) for k, v in ref.items() if k[0] == inter)

        # Check skip
        if out_csv.exists():
            existing = pd.read_csv(out_csv)
            if len(existing) >= expected_rows:
                frames_done_total += len(existing)
                inter_videos = sum(1 for k in ref if k[0] == inter)
                completed_videos_total += inter_videos
                print(f"  SKIP {inter}: already complete ({len(existing)} rows)", flush=True)
                continue

        # Build work list for this intersection
        video_dir = DATA_DIR / inter
        videos = sorted(video_dir.glob("*.mp4"))
        work = []
        for vp in videos:
            vname = vp.stem
            key = (inter, vname)
            if key not in ref:
                print(f"  WARNING: no detection ref for {inter}/{vname}, skipping", flush=True)
                continue
            expected_n = len(ref[key])
            work.append((inter, str(vp), str(FRAMES_DIR), expected_n, gabor_kernels))

        if not work:
            continue

        print(f"\n{'='*70}", flush=True)
        print(f"Processing {inter}: {len(work)} videos, {expected_rows} frames", flush=True)
        print(f"{'='*70}", flush=True)

        # Process videos one at a time: extract → process → delete frames
        # This keeps disk usage to ~1GB instead of ~24GB per intersection
        import shutil
        inter_results = []
        inter_frames_dir = FRAMES_DIR / inter

        for w in work:
            _, video_path_str, _, expected_n, _ = w
            vp = Path(video_path_str)
            vname = vp.stem
            vid_frames_dir = inter_frames_dir / vname
            vid_frames_dir.mkdir(parents=True, exist_ok=True)

            # 1. Extract frames for this video
            existing_f = sorted(glob.glob(str(vid_frames_dir / "frame_*.jpg")))
            if len(existing_f) < 1:
                print(f"  Extracting frames: {vname}...", flush=True)
                cmd = [
                    "ffmpeg", "-y", "-i", str(vp),
                    "-vf", "fps=2,scale=1280:720", "-q:v", "2",
                    str(vid_frames_dir / "frame_%06d.jpg")
                ]
                subprocess.run(cmd, capture_output=True, text=True)

            # 2. Process this video
            result = process_one_video(w)
            completed_videos_total += 1
            if result:
                inter_results.extend(result)
                frames_done_total += len(result)

            # 3. Delete frames immediately to save disk
            if vid_frames_dir.exists():
                shutil.rmtree(vid_frames_dir)

            elapsed = time.time() - t_start
            if frames_done_total > 0:
                fps_overall = frames_done_total / elapsed
                remaining = total_frames - frames_done_total
                eta_s = remaining / fps_overall if fps_overall > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta_s)))
            else:
                eta_str = "?"
            print(f"[{completed_videos_total}/{total_videos}] videos done, "
                  f"{frames_done_total}/{total_frames} frames, ETA: {eta_str}", flush=True)

        # Save intersection CSV
        if inter_results:
            _save_intersection(inter, inter_results, ref)

        # Clean up intersection frames dir if it still exists
        if inter_frames_dir.exists():
            shutil.rmtree(inter_frames_dir)
            print(f"  Cleaned up frames for {inter}", flush=True)

    _stop_autocommit.set()

    # 4. Concatenate all intersection CSVs
    print("\nConcatenating intersection CSVs...", flush=True)
    all_dfs = []
    for inter in INTERSECTIONS:
        csv_path = OUTPUT_DIR / f"features_30_{inter}.csv"
        if csv_path.exists():
            all_dfs.append(pd.read_csv(csv_path))
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged.to_csv(OUTPUT_DIR / "pod_new_30_features.csv", index=False)
        print(f"  Combined: {len(merged)} rows", flush=True)

        # Verification
        print("\n=== VERIFICATION ===", flush=True)
        print(f"Total rows: {len(merged)}", flush=True)
        nan_found = False
        for col in FEATURE_COLS:
            nan_ct = merged[col].isna().sum()
            if nan_ct > 0:
                print(f"  {col}: {nan_ct} NaN", flush=True)
                nan_found = True
        if not nan_found:
            print("  No NaN in any feature column!", flush=True)
        print(f"\nFeature statistics:", flush=True)
        for col in FEATURE_COLS:
            s = merged[col]
            print(f"  {col}: mean={s.mean():.4f} std={s.std():.4f} "
                  f"min={s.min():.4f} max={s.max():.4f}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {datetime.timedelta(seconds=int(elapsed_total))}", flush=True)
    print("Done!", flush=True)


def _save_intersection(inter_name, rows, ref):
    """Assign frame_ids from reference and save CSV."""
    df = pd.DataFrame(rows)
    # Assign frame_id from detection CSV reference
    frame_ids = []
    for _, row in df.iterrows():
        key = (row["sequence"], row["video"])
        idx = int(row["frame_index"])
        fids = ref.get(key, [])
        if idx < len(fids):
            frame_ids.append(fids[idx])
        else:
            frame_ids.append(-1)
    df["frame_id"] = frame_ids
    df = df.drop(columns=["frame_index"])
    # Reorder columns
    cols = ["sequence", "video", "frame_id"] + FEATURE_COLS
    df = df[cols]
    out_path = OUTPUT_DIR / f"features_30_{inter_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  SAVED {out_path} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
