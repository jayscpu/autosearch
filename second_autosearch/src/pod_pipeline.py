#!/usr/bin/env python3
"""ECHO Bellevue — Multi-intersection extraction pipeline.

Processes 5 traffic camera intersections from the Bellevue Traffic Video Dataset.
For each intersection:
  1. Extract frames at 2fps from each video file
  2. Run YOLO11n@640, YOLO11s@640, YOLO11m@640, YOLO11x@1280 (ground truth)
  3. Greedy IoU matching → fn_nano, fn_small, fn_medium per frame
  4. Extract spatial features (65 original + 10 new temporal/detector features)
  5. Output per-intersection CSVs

Checkpoints after each video file — safe to interrupt and resume.

Output:
  pod/pod_features_{intersection}.csv   — features + detections per frame
  pod/pod_detections_{intersection}.csv — raw detection results
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data"
DATA_DIR = SCRIPT_DIR / "data"
FRAMES_DIR = SCRIPT_DIR / "frames"
CKPT_DIR = SCRIPT_DIR / "checkpoints"

# ── Constants ──────────────────────────────────────────────────────
CONF = 0.25
IOU_THRESH = 0.5
VEHICLE_COCO = {2, 3, 5, 7}
FPS_TARGET = 2  # extract at 2fps

# Bellevue videos are 1280x720@30fps; we extract at original resolution for maximum ground truth quality
FRAME_W, FRAME_H = 1280, 720
STRIP_H = FRAME_H // 3

INTERSECTIONS = [
    "Bellevue_150th_Eastgate",
    "Bellevue_150th_Newport",
    "Bellevue_150th_SE38th",
    "Bellevue_Bellevue_NE8th",
    "Bellevue_116th_NE12th",
]

# Reusable OpenCV objects
DARK_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20)
FAST_DETECTOR_HALF = cv2.FastFeatureDetector_create(threshold=20)
MOG2_DICT = {}
_FFT_MASKS = {}


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Frame Extraction (per video file)
# ═══════════════════════════════════════════════════════════════════

def extract_frames_from_video(video_path, output_dir):
    """Extract frames at 2fps from a single video file.
    Returns list of frame file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if len(existing) > 0:
        return existing

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={FPS_TARGET},scale={FRAME_W}:{FRAME_H}",
        "-q:v", "2",
        str(output_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return sorted(output_dir.glob("frame_*.jpg"))


# ═══════════════════════════════════════════════════════════════════
# Phase 2: YOLO Inference
# ═══════════════════════════════════════════════════════════════════

YOLO_BATCH_SIZE = 8  # A100 80GB can handle 8 frames at once easily

def run_yolo_on_frames(frame_files, model, imgsz, label=""):
    """Run a YOLO model on a list of frame files using batch inference.
    Returns dict of frame_idx → boxes, confs."""
    dets = {}
    paths = [str(f) for f in frame_files]
    n = len(paths)

    for batch_start in range(0, n, YOLO_BATCH_SIZE):
        batch_end = min(batch_start + YOLO_BATCH_SIZE, n)
        batch_paths = paths[batch_start:batch_end]

        results = model.predict(batch_paths, conf=CONF, iou=IOU_THRESH,
                                imgsz=imgsz, verbose=False)

        for j, res in enumerate(results):
            idx = batch_start + j
            det = res.boxes
            boxes_all = det.xyxy.cpu().numpy()
            cls_all = det.cls.cpu().numpy().astype(int)
            conf_all = det.conf.cpu().numpy()

            mask = np.isin(cls_all, list(VEHICLE_COCO))
            dets[idx] = {
                "boxes": boxes_all[mask] if mask.any() else np.empty((0, 4)),
                "confs": conf_all[mask] if mask.any() else np.empty(0),
            }

        frames_done = min(batch_end, n)
        if frames_done % 500 < YOLO_BATCH_SIZE or frames_done == n:
            tag = f" ({label})" if label else ""
            print(f"      [{frames_done}/{n}]{tag}", flush=True)
    return dets


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Greedy IoU Matching
# ═══════════════════════════════════════════════════════════════════

def iou_matrix(a, b):
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-6)


def greedy_match(pred_boxes, gt_boxes):
    M, N = len(pred_boxes), len(gt_boxes)
    if M == 0 and N == 0:
        return 0, 0, 0, 0.0
    if M == 0:
        return 0, 0, N, 0.0
    if N == 0:
        return 0, M, 0, 0.0

    ious = iou_matrix(pred_boxes, gt_boxes)
    flat = np.argsort(ious.ravel())[::-1]
    matched_ious, used_p, used_g = [], set(), set()
    for idx in flat:
        i, j = divmod(int(idx), N)
        v = ious[i, j]
        if v < IOU_THRESH:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched_ious.append(v)

    tp = len(matched_ious)
    return tp, M - tp, N - tp, float(np.mean(matched_ious)) if matched_ious else 0.0


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def _get_fft_masks(h, w):
    key = (h, w)
    if key not in _FFT_MASKS:
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        r_nyquist = min(h, w) // 4
        r_half = min(h, w) // 8
        _FFT_MASKS[key] = {
            "low_nyquist": dist_sq <= r_nyquist ** 2,
            "very_low": dist_sq <= r_half ** 2,
        }
    return _FFT_MASKS[key]


def extract_resolution_features(gray, gray_f, bgr):
    h, w = gray.shape
    feats = {}

    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2
    total_energy = mag.sum() + 1e-10
    masks = _get_fft_masks(h, w)
    low_energy = mag[masks["low_nyquist"]].sum()
    feats["fft_high_freq_energy_ratio"] = float((total_energy - low_energy) / total_energy)

    very_low_energy = mag[masks["very_low"]].sum()
    mid_band_energy = low_energy - very_low_energy
    feats["fft_critical_band_ratio"] = float(mid_band_energy / total_energy)

    lap = cv2.Laplacian(gray_f, cv2.CV_64F)
    feats["laplacian_variance"] = float(lap.var())
    feats["laplacian_mean_abs"] = float(np.abs(lap).mean())

    edges_fine = cv2.Canny(gray, 50, 150)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3.0)
    edges_coarse = cv2.Canny(blurred, 50, 150)
    fine_density = edges_fine.mean() / 255.0
    coarse_density = edges_coarse.mean() / 255.0
    feats["edge_density_fine"] = float(fine_density)
    feats["edge_density_coarse"] = float(coarse_density)
    feats["edge_fine_coarse_ratio"] = float(fine_density / (coarse_density + 1e-8))

    kps_full = FAST_DETECTOR.detect(gray, None)
    n_kps_full = len(kps_full)
    feats["fast_keypoints_full"] = float(n_kps_full / (h * w))

    gray_half = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    kps_half = FAST_DETECTOR_HALF.detect(gray_half, None)
    n_kps_half = len(kps_half)
    feats["fast_keypoints_half"] = float(n_kps_half / (gray_half.shape[0] * gray_half.shape[1]))
    feats["keypoint_loss_ratio"] = float(1.0 - n_kps_half / (n_kps_full + 1e-8))

    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    sf = np.sqrt(np.mean(gx ** 2) + np.mean(gy ** 2))
    feats["spatial_frequency"] = float(sf)

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    feats["gradient_magnitude_mean"] = float(grad_mag.mean())
    feats["gradient_magnitude_std"] = float(grad_mag.std())

    dct_high_total = 0.0
    dct_total = 0.0
    n_blocks = 0
    step = 32
    for by in range(0, h - 7, step):
        for bx in range(0, w - 7, step):
            block = gray_f[by:by + 8, bx:bx + 8]
            dct_block = cv2.dct(block.astype(np.float32))
            energy = float(np.sum(dct_block ** 2))
            high_energy = float(np.sum(dct_block[4:, :] ** 2) +
                                np.sum(dct_block[:4, 4:] ** 2))
            dct_high_total += high_energy
            dct_total += energy
            n_blocks += 1
    feats["dct_high_freq_ratio"] = float(dct_high_total / (dct_total + 1e-10))

    gabor_energy = 0.0
    gray_gabor = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    gray_gabor_f = gray_gabor.astype(np.float64)
    for theta in [0, np.pi / 2]:
        kern = cv2.getGaborKernel((15, 15), sigma=2.5, theta=theta,
                                  lambd=4.0, gamma=0.5, psi=0)
        resp = cv2.filter2D(gray_gabor_f, cv2.CV_64F, kern)
        gabor_energy += np.mean(resp ** 2)
    feats["gabor_nyquist_energy"] = float(gabor_energy / 2.0)

    gray_half_up = cv2.resize(gray_half, (w, h), interpolation=cv2.INTER_LINEAR)
    mu1, mu2 = gray_f.mean(), gray_half_up.astype(np.float64).mean()
    sig1 = gray_f.std()
    sig2 = gray_half_up.astype(np.float64).std()
    sig12 = np.mean((gray_f - mu1) * (gray_half_up.astype(np.float64) - mu2))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 ** 2 + sig2 ** 2 + C2))
    feats["downsample_ssim"] = float(ssim_val)
    feats["downsample_info_loss"] = float(1.0 - ssim_val)

    return feats


def extract_density_features(gray, gray_f, bgr, seq_name, prev_gray):
    h, w = gray.shape
    feats = {}

    if seq_name not in MOG2_DICT:
        mog = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True)
        MOG2_DICT[seq_name] = mog
    else:
        mog = MOG2_DICT[seq_name]

    fg_mask = mog.apply(bgr)
    fg_binary = (fg_mask == 255).astype(np.uint8)
    shadow_binary = (fg_mask == 127).astype(np.uint8)
    total_px = h * w

    feats["foreground_pixel_ratio"] = float(fg_binary.sum() / total_px)
    feats["shadow_pixel_ratio"] = float(shadow_binary.sum() / total_px)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_clean = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_clean, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = (areas >= 200) & (areas < 0.3 * total_px)
        feats["foreground_blob_count"] = int(valid.sum())
        if valid.sum() > 0:
            valid_areas = areas[valid]
            feats["foreground_blob_area_mean"] = float(valid_areas.mean() / total_px)
            feats["foreground_blob_area_std"] = float(valid_areas.std() / total_px)
            perimeters = stats[1:, cv2.CC_STAT_WIDTH] * 2 + stats[1:, cv2.CC_STAT_HEIGHT] * 2
            valid_pa = perimeters[valid].astype(float) / (valid_areas + 1e-8)
            feats["foreground_blob_pa_ratio"] = float(valid_pa.mean())
        else:
            feats["foreground_blob_area_mean"] = 0.0
            feats["foreground_blob_area_std"] = 0.0
            feats["foreground_blob_pa_ratio"] = 0.0
    else:
        feats["foreground_blob_count"] = 0
        feats["foreground_blob_area_mean"] = 0.0
        feats["foreground_blob_area_std"] = 0.0
        feats["foreground_blob_pa_ratio"] = 0.0

    edges = cv2.Canny(gray, 50, 150)
    fg_edges = cv2.bitwise_and(edges, edges, mask=fg_clean)
    feats["foreground_edge_density"] = float(fg_edges.sum() / (255.0 * total_px))

    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        feats["temporal_diff_mean"] = float(diff.mean())
        feats["temporal_diff_std"] = float(diff.astype(np.float64).std())
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        feats["motion_pixel_ratio"] = float(motion_mask.sum() / (255.0 * total_px))
    else:
        feats["temporal_diff_mean"] = 0.0
        feats["temporal_diff_std"] = 0.0
        feats["motion_pixel_ratio"] = 0.0

    return feats


def extract_complexity_features(gray, gray_f, bgr):
    h, w = gray.shape
    feats = {}

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-10)
    feats["image_entropy"] = float(scipy_entropy(hist_norm, base=2))

    gray_ds = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    gray_q = (gray_ds // 8).astype(np.int32)
    levels = 32
    glcm = np.zeros((levels, levels), dtype=np.float64)
    left, right = gray_q[:, :-1].ravel(), gray_q[:, 1:].ravel()
    np.add.at(glcm, (left, right), 1)
    np.add.at(glcm, (right, left), 1)
    top_p, bot_p = gray_q[:-1, :].ravel(), gray_q[1:, :].ravel()
    np.add.at(glcm, (top_p, bot_p), 1)
    np.add.at(glcm, (bot_p, top_p), 1)
    glcm_sum = glcm.sum() + 1e-10
    glcm_norm = glcm / glcm_sum

    ii, jj = np.meshgrid(range(levels), range(levels), indexing='ij')
    feats["glcm_contrast"] = float(np.sum(glcm_norm * (ii - jj) ** 2))
    feats["glcm_homogeneity"] = float(np.sum(glcm_norm / (1 + np.abs(ii - jj))))
    feats["glcm_energy"] = float(np.sum(glcm_norm ** 2))
    glcm_flat = glcm_norm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]
    feats["glcm_entropy"] = float(-np.sum(glcm_flat * np.log2(glcm_flat)))

    B, G, R = bgr[:, :, 0].astype(float), bgr[:, :, 1].astype(float), bgr[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    feats["colorfulness"] = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
        0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    feats["mean_brightness"] = float(gray.mean())
    feats["brightness_std"] = float(gray_f.std())

    min_channel = np.min(bgr, axis=2)
    dark = cv2.erode(min_channel, DARK_KERNEL)
    feats["dark_channel_mean"] = float(dark.mean())

    edges = cv2.Canny(gray, 50, 150)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        feats["small_edge_components"] = int(np.sum(areas < 500))
        feats["total_edge_components"] = int(n_labels - 1)
    else:
        feats["small_edge_components"] = 0
        feats["total_edge_components"] = 0

    feats["overall_edge_density"] = float(edges.mean() / 255.0)
    feats["rms_contrast"] = float(np.sqrt(np.mean((gray_f - gray_f.mean()) ** 2)))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)
    feats["saturation_mean"] = float(sat.mean())
    feats["saturation_std"] = float(sat.std())

    return feats


def extract_strip_features(gray):
    strips = {
        "top": gray[0:STRIP_H, :],
        "mid": gray[STRIP_H:2 * STRIP_H, :],
        "bot": gray[2 * STRIP_H:FRAME_H, :],
    }
    feats = {}
    strip_vals = {}
    for prefix, strip in strips.items():
        gray_f = strip.astype(np.float64)
        lap_var = float(cv2.Laplacian(gray_f, cv2.CV_64F).var())
        feats[f"{prefix}_laplacian_var"] = lap_var

        gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
        grad_std = float(np.sqrt(gx ** 2 + gy ** 2).std())
        feats[f"{prefix}_gradient_std"] = grad_std

        edges = cv2.Canny(strip, 50, 150)
        ed = float(edges.mean() / 255.0)
        feats[f"{prefix}_edge_density"] = ed

        h_s, w_s = strip.shape
        f = np.fft.fft2(gray_f)
        fshift = np.fft.fftshift(f)
        mag_s = np.abs(fshift) ** 2
        total_e = mag_s.sum() + 1e-10
        strip_masks = _get_fft_masks(h_s, w_s)
        low_e = mag_s[strip_masks["very_low"]].sum()
        feats[f"{prefix}_high_freq_energy"] = float((total_e - low_e) / total_e)

        strip_vals[prefix] = {"edge_density": ed, "gradient_std": grad_std,
                              "laplacian_var": lap_var}

    for metric in ["edge_density", "gradient_std", "laplacian_var"]:
        top_v = strip_vals["top"][metric]
        bot_v = strip_vals["bot"][metric]
        feats[f"ratio_top_bot_{metric}"] = float(top_v / (bot_v + 1e-8))

    return feats


def extract_brisque_features(gray_f):
    feats = {}
    kernel_size = 7
    mu = cv2.GaussianBlur(gray_f, (kernel_size, kernel_size), 1.166)
    mu_sq = cv2.GaussianBlur(gray_f ** 2, (kernel_size, kernel_size), 1.166)
    sigma = np.sqrt(np.abs(mu_sq - mu ** 2))
    mscn = (gray_f - mu) / (sigma + 1.0)

    feats["mscn_mean"] = float(mscn.mean())
    feats["mscn_std"] = float(mscn.std())
    feats["mscn_kurtosis"] = float(
        np.mean((mscn - mscn.mean()) ** 4) / (mscn.std() ** 4 + 1e-10) - 3.0)
    feats["mscn_skewness"] = float(
        np.mean((mscn - mscn.mean()) ** 3) / (mscn.std() ** 3 + 1e-10))

    h_pairs = mscn[:, :-1] * mscn[:, 1:]
    feats["mscn_h_pair_mean"] = float(h_pairs.mean())
    feats["mscn_h_pair_std"] = float(h_pairs.std())

    v_pairs = mscn[:-1, :] * mscn[1:, :]
    feats["mscn_v_pair_mean"] = float(v_pairs.mean())
    feats["mscn_v_pair_std"] = float(v_pairs.std())

    return feats


# ── New features (Bellevue additions) ──────────────────────────────

def extract_optical_flow_features(gray, prev_gray):
    """Farneback optical flow features."""
    feats = {}
    if prev_gray is None:
        feats["optical_flow_magnitude_mean"] = 0.0
        feats["optical_flow_magnitude_std"] = 0.0
        feats["optical_flow_direction_entropy"] = 0.0
        return feats

    # Downsample for speed (~15ms instead of ~60ms at full res)
    h, w = gray.shape
    small_gray = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_prev = cv2.resize(prev_gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    flow = cv2.calcOpticalFlowFarneback(
        small_prev, small_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    feats["optical_flow_magnitude_mean"] = float(mag.mean())
    feats["optical_flow_magnitude_std"] = float(mag.std())

    # Direction entropy: bin angles into 8 bins, compute entropy
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    # Only count pixels with meaningful motion
    motion_mask = mag > 1.0
    if motion_mask.sum() > 100:
        angles_moving = angle[motion_mask]
        hist, _ = np.histogram(angles_moving, bins=8, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-10)
        feats["optical_flow_direction_entropy"] = float(scipy_entropy(hist, base=2))
    else:
        feats["optical_flow_direction_entropy"] = 0.0

    return feats


def extract_ssim_feature(gray_f, prev_gray_f):
    """SSIM between current and previous frame — measures scene change."""
    feats = {}
    if prev_gray_f is None:
        feats["ssim_vs_prev"] = 1.0
        return feats

    mu1, mu2 = gray_f.mean(), prev_gray_f.mean()
    sig1, sig2 = gray_f.std(), prev_gray_f.std()
    sig12 = np.mean((gray_f - mu1) * (prev_gray_f - mu2))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 ** 2 + sig2 ** 2 + C2))
    feats["ssim_vs_prev"] = float(ssim)
    return feats


def extract_detector_features(nano_det):
    """Features derived from YOLO11n detections (essentially free)."""
    feats = {}
    boxes = nano_det["boxes"]
    confs = nano_det["confs"]
    n = len(boxes)

    feats["det_count_nano"] = float(n)

    if n == 0:
        feats["det_confidence_mean"] = 0.0
        feats["det_spatial_dispersion"] = 0.0
        feats["det_small_object_ratio"] = 0.0
        return feats

    feats["det_confidence_mean"] = float(confs.mean())

    # Spatial dispersion: std of bbox centroids
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    feats["det_spatial_dispersion"] = float(np.sqrt(cx.std() ** 2 + cy.std() ** 2))

    # Small object ratio: boxes with area < 2% of frame
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    frame_area = FRAME_W * FRAME_H
    small_threshold = 0.02 * frame_area
    feats["det_small_object_ratio"] = float((areas < small_threshold).sum() / n)

    return feats


def extract_small_detector_features(small_det):
    """Features derived from YOLO11s detections."""
    feats = {}
    boxes = small_det["boxes"]
    confs = small_det["confs"]
    n = len(boxes)
    feats["det_count_small"] = float(n)
    feats["det_confidence_small"] = float(confs.mean()) if n > 0 else 0.0
    return feats


def extract_all_features(bgr, prev_gray, seq_name, nano_det=None, small_det=None):
    """Extract all features for a single frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)
    prev_gray_f = prev_gray.astype(np.float64) if prev_gray is not None else None

    feats = {}
    feats.update(extract_resolution_features(gray, gray_f, bgr))
    feats.update(extract_density_features(gray, gray_f, bgr, seq_name, prev_gray))
    feats.update(extract_complexity_features(gray, gray_f, bgr))
    feats.update(extract_strip_features(gray))
    feats.update(extract_brisque_features(gray_f))
    # New features
    feats.update(extract_optical_flow_features(gray, prev_gray))
    feats.update(extract_ssim_feature(gray_f, prev_gray_f))
    if nano_det is not None:
        feats.update(extract_detector_features(nano_det))
    else:
        feats.update({"det_count_nano": 0.0, "det_confidence_mean": 0.0,
                      "det_spatial_dispersion": 0.0, "det_small_object_ratio": 0.0})
    if small_det is not None:
        feats.update(extract_small_detector_features(small_det))
    else:
        feats.update({"det_count_small": 0.0, "det_confidence_small": 0.0})

    return feats, gray


# ═══════════════════════════════════════════════════════════════════
# Main Processing Loop (per-video checkpointing)
# ═══════════════════════════════════════════════════════════════════

def process_intersection(intersection_name):
    """Process all videos for one intersection. Checkpoints per video."""
    video_dir = DATA_DIR / intersection_name
    if not video_dir.exists():
        print(f"  SKIP {intersection_name}: data directory not found")
        return None

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print(f"  SKIP {intersection_name}: no .mp4 files found")
        return None

    print(f"\n{'='*70}")
    print(f"Processing: {intersection_name} ({len(videos)} videos)")
    print(f"{'='*70}")

    # Checkpoint file: tracks which videos are done
    ckpt_file = CKPT_DIR / f"ckpt_{intersection_name}.json"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = json.load(f)
    else:
        ckpt = {"completed_videos": [], "global_frame_id": 0}

    # Output CSVs
    features_csv = OUTPUT_DIR / f"pod_features_{intersection_name}.csv"
    dets_csv = OUTPUT_DIR / f"pod_detections_{intersection_name}.csv"

    # Load existing partial results
    all_feat_records = []
    all_det_records = []
    if features_csv.exists():
        existing_feats = pd.read_csv(features_csv)
        all_feat_records = existing_feats.to_dict("records")
    if dets_csv.exists():
        existing_dets = pd.read_csv(dets_csv)
        all_det_records = existing_dets.to_dict("records")

    # Load YOLO models once
    from ultralytics import YOLO
    print("  Loading YOLO models ...")
    nano_model = YOLO("yolo11n.pt")
    small_model = YOLO("yolo11s.pt")
    medium_model = YOLO("yolo11m.pt")
    x_model = YOLO("yolo11x.pt")

    global_fid = ckpt["global_frame_id"]

    for video_path in videos:
        vname = video_path.stem
        if vname in ckpt["completed_videos"]:
            print(f"  SKIP {vname} (already completed)")
            continue

        print(f"\n  Processing video: {vname}")
        t0 = time.time()

        # Phase 1: Extract frames
        vid_frames_dir = FRAMES_DIR / intersection_name / vname
        frame_files = extract_frames_from_video(video_path, vid_frames_dir)
        n_frames = len(frame_files)
        print(f"    Frames: {n_frames}")

        if n_frames == 0:
            print(f"    WARNING: No frames extracted, skipping")
            ckpt["completed_videos"].append(vname)
            continue

        # Phase 2: YOLO inference
        print(f"    Running YOLO11n@640 ...", flush=True)
        nano_dets = run_yolo_on_frames(frame_files, nano_model, 640, label="nano@640")
        print(f"    Running YOLO11s@640 ...", flush=True)
        small_dets = run_yolo_on_frames(frame_files, small_model, 640, label="small@640")
        print(f"    Running YOLO11m@640 ...", flush=True)
        medium_dets = run_yolo_on_frames(frame_files, medium_model, 640, label="medium@640")
        print(f"    Running YOLO11x@1280 ...", flush=True)
        x_dets = run_yolo_on_frames(frame_files, x_model, 1280, label="x@1280")

        # Phase 3+4: Matching + Feature extraction
        print(f"    Matching + feature extraction ...", flush=True)
        prev_gray = None
        seq_name = f"{intersection_name}_{vname}"

        for i in range(n_frames):
            global_fid += 1
            fpath = frame_files[i]
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                continue

            # Match nano, small, and medium vs x
            nano_boxes = nano_dets[i]["boxes"]
            small_boxes = small_dets[i]["boxes"]
            medium_boxes = medium_dets[i]["boxes"]
            x_boxes = x_dets[i]["boxes"]

            n_tp, n_fp, n_fn, _ = greedy_match(nano_boxes, x_boxes)
            s_tp, s_fp, s_fn, _ = greedy_match(small_boxes, x_boxes)
            m_tp, m_fp, m_fn, _ = greedy_match(medium_boxes, x_boxes)

            det_record = {
                "sequence": intersection_name,
                "video": vname,
                "frame_id": global_fid,
                "fn_nano": n_fn, "nano_tp": n_tp, "nano_fp": n_fp,
                "nano_count": len(nano_boxes),
                "fn_small": s_fn, "small_tp": s_tp, "small_fp": s_fp,
                "small_count": len(small_boxes),
                "fn_medium": m_fn, "medium_tp": m_tp, "medium_fp": m_fp,
                "medium_count": len(medium_boxes),
                "x_count": len(x_boxes),
            }
            all_det_records.append(det_record)

            # Extract features
            feats, gray = extract_all_features(bgr, prev_gray, seq_name, nano_dets[i], small_dets[i])
            prev_gray = gray

            feats["sequence"] = intersection_name
            feats["video"] = vname
            feats["frame_id"] = global_fid
            feats["fn_nano"] = n_fn
            feats["fn_small"] = s_fn
            feats["fn_medium"] = m_fn
            feats["x_count"] = len(x_boxes)
            feats["nano_tp"] = n_tp
            feats["nano_fp"] = n_fp
            feats["small_tp"] = s_tp
            feats["small_fp"] = s_fp
            feats["medium_tp"] = m_tp
            feats["medium_fp"] = m_fp
            all_feat_records.append(feats)

            if (i + 1) % 500 == 0 or i == n_frames - 1:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                eta = (n_frames - i - 1) / fps / 60
                print(f"      [{i+1:5d}/{n_frames}] {fps:.1f} fps, "
                      f"ETA {eta:.1f} min", flush=True)

        # Save after each video (checkpoint)
        pd.DataFrame(all_feat_records).to_csv(features_csv, index=False)
        pd.DataFrame(all_det_records).to_csv(dets_csv, index=False)

        ckpt["completed_videos"].append(vname)
        ckpt["global_frame_id"] = global_fid
        with open(ckpt_file, "w") as f:
            json.dump(ckpt, f)

        elapsed = time.time() - t0
        print(f"    Done: {n_frames} frames in {elapsed:.0f}s "
              f"({n_frames/elapsed:.1f} fps). Checkpoint saved.")

        # Clean up frames to save disk
        # (comment this out if you want to keep frames for debugging)
        import shutil
        shutil.rmtree(vid_frames_dir, ignore_errors=True)

    # Reset MOG2 for next intersection
    keys_to_del = [k for k in MOG2_DICT if k.startswith(intersection_name)]
    for k in keys_to_del:
        del MOG2_DICT[k]

    print(f"\n  {intersection_name}: COMPLETE — {global_fid} total frames")
    return features_csv


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ECHO Bellevue extraction pipeline")
    parser.add_argument("--intersection", type=str, default=None,
                        help="Process only this intersection (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("ECHO BELLEVUE — Multi-Intersection Extraction Pipeline")
    print("=" * 70)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    if args.intersection:
        targets = [args.intersection]
    else:
        targets = INTERSECTIONS

    for name in targets:
        process_intersection(name)

    # Skip merge when processing a single intersection
    if len(targets) == 1:
        print("\nDone (single intersection, skipping merge).")
        return

    # Merge all intersection CSVs into one combined file
    print("\n" + "=" * 70)
    print("MERGING ALL INTERSECTIONS")
    print("=" * 70)

    all_dfs = []
    for name in INTERSECTIONS:
        csv_path = OUTPUT_DIR / f"pod_features_{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            print(f"  {name}: {len(df)} frames")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = OUTPUT_DIR / "pod_features_all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined: {len(combined)} frames → {combined_path}")

        # Print summary
        for name in INTERSECTIONS:
            sub = combined[combined["sequence"] == name]
            if len(sub) > 0:
                mr = sub["fn_nano"] / sub["x_count"].clip(lower=1)
                print(f"    {name}: {len(sub)} frames, "
                      f"mean fn_nano={sub['fn_nano'].mean():.2f}, "
                      f"mean miss_rate={mr.mean():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
