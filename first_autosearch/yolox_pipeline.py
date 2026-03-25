#!/usr/bin/env python3
"""Fresno traffic camera experiment: YOLO11n@320 & YOLO11m@640 vs YOLO11x@640.

Single 3-hour OBS recording of a Fresno traffic camera.
Pure temporal split: train on first 2 hours, validate on last 1 hour.

Pipeline:
  1. Extract frames at 2fps (960x540)
  2. Run YOLO11n@320, YOLO11m@640 (cheap models), YOLO11x@640 (pseudo GT)
  3. Greedy IoU matching: nano vs x → fn_nano, medium vs x → fn_medium
  4. Extract 65 spatial features (same as DETRAC pipeline)
  5. Build shifted-label windows, temporal split
  6. Train RF + LSTM for each target (fn_nano, fn_medium)
  7. Evaluate + PDF report

Output:
  yolox_detections.csv
  yolox_features.csv
  yolox_results.csv
  yolox_results.pdf
"""

import pickle
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import transition_accuracy

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FRESNO_DIR = SCRIPT_DIR.parent
ECHO_ROOT = Path("/home/aljouhara/Desktop/git-echo/echo")

VIDEO = FRESNO_DIR / "fresno_trimmed.mp4"
FRAMES_DIR = FRESNO_DIR / "frames"
DETS_PKL = SCRIPT_DIR / "yolox_detections.pkl"
DETS_CSV = SCRIPT_DIR / "yolox_detections.csv"
FEATURES_CSV = SCRIPT_DIR / "yolox_features.csv"
OUT_PDF = SCRIPT_DIR / "yolox_results.pdf"
OUT_CSV = SCRIPT_DIR / "yolox_results.csv"

# ── Constants ──────────────────────────────────────────────────────
CONF = 0.25
IOU_THRESH = 0.5
VEHICLE_COCO = {2, 3, 5, 7}

FRAME_H, FRAME_W = 540, 960
STRIP_H = FRAME_H // 3

WINDOW = 30
HORIZON = 30
TRAIN_STRIDE = 15
EVAL_STRIDE = 30
N_CLASSES = 2
KL_ANNEALING_EPOCHS = 10

# First 2 hours at 2fps = 7200s * 2 = 14400 frames
TRAIN_CUTOFF = 14400

SELECTED_FEATURES = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

# Reusable OpenCV objects
DARK_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20)
FAST_DETECTOR_320 = cv2.FastFeatureDetector_create(threshold=20)
MOG2_DICT = {}
_FFT_MASKS = {}


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Frame Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_frames():
    """Extract frames at 2fps, downscaled to 960x540."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if len(existing) >= 21000:
        print(f"  Phase 1 SKIP: {len(existing)} frames already exist.")
        return len(existing)

    print("  Phase 1: Extracting frames at 2fps, 960x540 ...", flush=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(VIDEO),
        "-vf", "fps=2,scale=960:540",
        "-q:v", "2",
        str(FRAMES_DIR / "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    n_frames = len(sorted(FRAMES_DIR.glob("frame_*.jpg")))
    print(f"  Phase 1 DONE: {n_frames} frames extracted.")
    return n_frames


# ═══════════════════════════════════════════════════════════════════
# Phase 2: YOLO Inference
# ═══════════════════════════════════════════════════════════════════

def run_yolo_inference(n_frames):
    """Run YOLO11n@320, YOLO11m@640, and YOLO11x@640 on all frames."""
    all_dets = {}
    if DETS_PKL.exists():
        with open(DETS_PKL, "rb") as f:
            all_dets = pickle.load(f)

    from ultralytics import YOLO

    frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    n = len(frame_files)

    models = [
        {"name": "nano", "file": str(ECHO_ROOT / "yolo11n.pt"), "imgsz": 320},
        {"name": "medium", "file": str(ECHO_ROOT / "yolo11m.pt"), "imgsz": 640},
        {"name": "x", "file": str(SCRIPT_DIR / "yolo11x.pt"), "imgsz": 640},
    ]

    # Skip models already in the pickle
    models = [m for m in models if m["name"] not in all_dets]
    if not models:
        print("  Phase 2 SKIP: All model detections already exist.")
        return all_dets

    for m in models:
        print(f"\n  Phase 2: Running YOLO11{m['name']}@{m['imgsz']} on {n} frames ...",
              flush=True)
        model = YOLO(m["file"])
        dets = {}
        t0 = time.time()

        for i, fpath in enumerate(frame_files):
            frame_id = i + 1
            results = model.predict(str(fpath), conf=CONF, iou=IOU_THRESH,
                                    imgsz=m["imgsz"], verbose=False)
            det = results[0].boxes
            boxes_all = det.xyxy.cpu().numpy()
            cls_all = det.cls.cpu().numpy().astype(int)

            # Filter to vehicle classes
            mask = np.isin(cls_all, list(VEHICLE_COCO))
            dets[frame_id] = boxes_all[mask] if mask.any() else np.empty((0, 4))

            if (i + 1) % 2000 == 0 or i == n - 1:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                eta = (n - i - 1) / fps / 60
                print(f"    [{i+1:6d}/{n}] {fps:.1f} fps, ETA {eta:.1f} min",
                      flush=True)

        all_dets[m["name"]] = dets
        elapsed = time.time() - t0
        print(f"  YOLO11{m['name']}@{m['imgsz']} done in {elapsed:.0f}s "
              f"({n/elapsed:.1f} fps)")

    with open(DETS_PKL, "wb") as f:
        pickle.dump(all_dets, f)
    print(f"  Phase 2 DONE: Saved to {DETS_PKL}")
    return all_dets


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Greedy IoU Matching (copied from res_evaluate_detrac.py)
# ═══════════════════════════════════════════════════════════════════

def iou_matrix(a, b):
    """Vectorized IoU between (M,4) and (N,4) boxes -> (M,N)."""
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
    """Class-agnostic greedy matching. Returns (tp, fp, fn, mean_iou_of_tps)."""
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


def compute_matching(all_dets):
    """Match nano and medium detections against x detections for all frames."""
    # Check if existing CSV already has fn_medium
    if DETS_CSV.exists():
        existing = pd.read_csv(DETS_CSV)
        if "fn_medium" in existing.columns:
            print("  Phase 3 SKIP: Detections CSV already has fn_nano + fn_medium.")
            return existing
        print("  Phase 3: CSV exists but missing fn_medium. Recomputing ...")

    print("  Phase 3: Computing greedy IoU matching (nano & medium vs x) ...",
          flush=True)
    nano_dets = all_dets["nano"]
    medium_dets = all_dets["medium"]
    x_dets = all_dets["x"]

    records = []
    frame_ids = sorted(x_dets.keys())
    for fid in frame_ids:
        nano_boxes = nano_dets[fid]
        medium_boxes = medium_dets[fid]
        x_boxes = x_dets[fid]

        n_tp, n_fp, n_fn, n_miou = greedy_match(nano_boxes, x_boxes)
        m_tp, m_fp, m_fn, m_miou = greedy_match(medium_boxes, x_boxes)

        records.append({
            "frame_id": fid,
            "fn_nano": n_fn,
            "nano_tp": n_tp,
            "nano_fp": n_fp,
            "nano_count": len(nano_boxes),
            "fn_medium": m_fn,
            "medium_tp": m_tp,
            "medium_fp": m_fp,
            "medium_count": len(medium_boxes),
            "x_count": len(x_boxes),
            "mean_iou_nano": n_miou,
            "mean_iou_medium": m_miou,
        })

    df = pd.DataFrame(records)
    df.to_csv(DETS_CSV, index=False)
    print(f"  Phase 3 DONE: {len(df)} frames")
    print(f"    mean fn_nano={df['fn_nano'].mean():.2f}, "
          f"mean fn_medium={df['fn_medium'].mean():.2f}, "
          f"mean x_count={df['x_count'].mean():.1f}")
    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Feature Extraction (copied from new_extract_features.py)
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
    kps_half = FAST_DETECTOR_320.detect(gray_half, None)
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
            history=200, varThreshold=50, detectShadows=True
        )
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
        fg_clean, connectivity=8
    )
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
        0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

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
        np.mean((mscn - mscn.mean()) ** 4) / (mscn.std() ** 4 + 1e-10) - 3.0
    )
    feats["mscn_skewness"] = float(
        np.mean((mscn - mscn.mean()) ** 3) / (mscn.std() ** 3 + 1e-10)
    )

    h_pairs = mscn[:, :-1] * mscn[:, 1:]
    feats["mscn_h_pair_mean"] = float(h_pairs.mean())
    feats["mscn_h_pair_std"] = float(h_pairs.std())

    v_pairs = mscn[:-1, :] * mscn[1:, :]
    feats["mscn_v_pair_mean"] = float(v_pairs.mean())
    feats["mscn_v_pair_std"] = float(v_pairs.std())

    return feats


def extract_all_features(bgr, prev_gray, seq_name):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)
    feats = {}
    feats.update(extract_resolution_features(gray, gray_f, bgr))
    feats.update(extract_density_features(gray, gray_f, bgr, seq_name, prev_gray))
    feats.update(extract_complexity_features(gray, gray_f, bgr))
    feats.update(extract_strip_features(gray))
    feats.update(extract_brisque_features(gray_f))
    return feats, gray


def run_feature_extraction(det_df):
    """Extract 65 features for all frames."""
    if FEATURES_CSV.exists():
        existing = pd.read_csv(FEATURES_CSV)
        if "fn_medium" in existing.columns:
            print("  Phase 4 SKIP: Features CSV already has fn_medium.")
            return existing
        # Features exist but missing fn_medium — merge from det_df
        print("  Phase 4: Merging fn_medium into existing features CSV ...")
        med_cols = det_df[["frame_id", "fn_medium", "medium_tp", "medium_fp",
                           "medium_count"]].copy()
        existing = existing.merge(med_cols, on="frame_id", how="left")
        existing["fn_medium"] = existing["fn_medium"].fillna(0).astype(int)
        existing["medium_tp"] = existing["medium_tp"].fillna(0).astype(int)
        existing["medium_fp"] = existing["medium_fp"].fillna(0).astype(int)
        existing["medium_count"] = existing["medium_count"].fillna(0).astype(int)
        existing.to_csv(FEATURES_CSV, index=False)
        print(f"  Phase 4 DONE: Merged fn_medium into {len(existing)} rows.")
        return existing

    print("  Phase 4: Extracting 65 spatial features ...", flush=True)
    frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    n = len(frame_files)

    # Get feature column names from dummy extraction
    test_img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    test_feats, _ = extract_all_features(test_img, None, "__test__")
    feat_cols = sorted(test_feats.keys())
    del MOG2_DICT["__test__"]

    meta_cols = ["sequence", "frame_id", "fn_nano", "fn_medium", "x_count",
                 "nano_tp", "nano_fp", "medium_tp", "medium_fp"]
    col_order = meta_cols + feat_cols
    print(f"    {len(feat_cols)} features per frame")

    # Index detection df by frame_id for fast lookup
    det_lookup = det_df.set_index("frame_id")

    prev_gray = None
    records = []
    t0 = time.time()

    for i, fpath in enumerate(frame_files):
        frame_id = i + 1
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            print(f"    WARNING: Could not read {fpath}", flush=True)
            continue

        feats, gray = extract_all_features(bgr, prev_gray, "fresno")
        prev_gray = gray

        # Add metadata
        feats["sequence"] = "fresno"
        feats["frame_id"] = frame_id
        if frame_id in det_lookup.index:
            row = det_lookup.loc[frame_id]
            feats["fn_nano"] = int(row["fn_nano"])
            feats["fn_medium"] = int(row["fn_medium"])
            feats["x_count"] = int(row["x_count"])
            feats["nano_tp"] = int(row["nano_tp"])
            feats["nano_fp"] = int(row["nano_fp"])
            feats["medium_tp"] = int(row["medium_tp"])
            feats["medium_fp"] = int(row["medium_fp"])
        else:
            feats["fn_nano"] = 0
            feats["fn_medium"] = 0
            feats["x_count"] = 0
            feats["nano_tp"] = 0
            feats["nano_fp"] = 0
            feats["medium_tp"] = 0
            feats["medium_fp"] = 0

        records.append(feats)

        if (i + 1) % 2000 == 0 or i == n - 1:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            eta = (n - i - 1) / fps / 60
            print(f"    [{i+1:6d}/{n}] {fps:.1f} fps, ETA {eta:.1f} min", flush=True)

    df = pd.DataFrame(records)[col_order]
    df.to_csv(FEATURES_CSV, index=False)
    print(f"  Phase 4 DONE: {len(df)} frames, {len(feat_cols)} features.")
    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Windowing & Temporal Split
# ═══════════════════════════════════════════════════════════════════

def build_windows(df, feature_cols, target_col, stride, scaler=None,
                  threshold=None):
    total_need = WINDOW + HORIZON
    X_list, y_list, y_raw_list, seq_ids = [], [], [], []

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        feat_vals = sdf[feature_cols].values
        target_vals = sdf[target_col].values

        if scaler is not None:
            feat_vals = scaler.transform(feat_vals)

        for t in range(0, n - total_need + 1, stride):
            x_window = feat_vals[t : t + WINDOW]
            target_mean = target_vals[t + WINDOW : t + WINDOW + HORIZON].mean()

            X_list.append(x_window)
            y_list.append(int(target_mean >= threshold))
            y_raw_list.append(target_mean)
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    y_raw = np.array(y_raw_list, dtype=np.float32)
    return X, y, y_raw, seq_ids


def build_rf_features(X_windows):
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()
    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64),
                       t_centered) / t_var
    return np.hstack([means, stds, slopes.astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# Phase 6: LSTM + Training
# ═══════════════════════════════════════════════════════════════════

class PlainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        return probs.argmax(dim=1), probs[:, 1], entropy


def train_lstm(X_train, y_train, X_val, y_val, n_feat, device):
    model = PlainLSTM(input_size=n_feat).to(device)
    criterion = nn.CrossEntropyLoss()
    print(f"  LSTM params: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=128)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_acc, best_state, patience = 0.0, None, 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(200):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += len(yb)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += len(yb)

        ta, va = t_correct / t_total, v_correct / v_total
        tl, vl = t_loss / t_total, v_loss / v_total
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step(va)

        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or patience == 0:
            m = " *" if patience == 0 else ""
            print(f"  Epoch {epoch:3d}: acc={ta:.3f}/{va:.3f} "
                  f"loss={tl:.4f}/{vl:.4f}{m}", flush=True)
        if patience >= 25:
            print(f"  Early stop epoch {epoch} (best={best_acc:.4f})")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, history


# ═══════════════════════════════════════════════════════════════════
# Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════

def eval_clf(y_true, y_pred, y_prob=None, label=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {label:40s} Acc={acc:.3f} F1={f1:.3f} "
          f"Prec={prec:.3f} Rec={rec:.3f} AUC={auc:.3f}")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc, "cm": cm}


# transition_accuracy() imported from utils.py


# ═══════════════════════════════════════════════════════════════════
# Phase 7: Full Evaluation + PDF Report
# ═══════════════════════════════════════════════════════════════════

def run_single_target(features_df, target_col, target_label, device):
    """Train models and evaluate for a single target (fn_nano or fn_medium).
    Returns (results_dict, model_artifacts) for PDF generation."""
    print(f"\n{'#' * 70}")
    print(f"# TARGET: {target_label} ({target_col})")
    print(f"{'#' * 70}")

    # ── Temporal Split ──
    train_df = features_df[features_df["frame_id"] <= TRAIN_CUTOFF].copy()
    val_df = features_df[features_df["frame_id"] > TRAIN_CUTOFF].copy()
    print(f"  Train frames: {len(train_df):,} | Val frames: {len(val_df):,}")

    # ── Threshold (train only — avoid data leakage) ──
    all_targets = []
    for _, sdf in train_df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        for t in range(0, len(sdf) - WINDOW - HORIZON + 1, EVAL_STRIDE):
            all_targets.append(sdf[target_col].iloc[t + WINDOW:t + WINDOW + HORIZON].mean())
    threshold = float(np.median(all_targets))
    if threshold <= 0.0:
        threshold = 0.01
        print(f"  Threshold adjusted to {threshold:.4f} (median was 0)")
    else:
        print(f"  Threshold (median {target_col}): {threshold:.4f}")

    all_targets_arr = np.array(all_targets)
    hard_rate = (all_targets_arr >= threshold).mean()
    print(f"  Class balance: {100*hard_rate:.1f}% hard / {100*(1-hard_rate):.1f}% easy")

    # ── Scaler ──
    scaler = StandardScaler()
    scaler.fit(train_df[SELECTED_FEATURES].values)

    # ── Build windows ──
    X_train, y_train, _, train_ids = build_windows(
        train_df, SELECTED_FEATURES, target_col, TRAIN_STRIDE, scaler, threshold)
    X_val, y_val, y_raw_val, val_ids = build_windows(
        val_df, SELECTED_FEATURES, target_col, EVAL_STRIDE, scaler, threshold)

    print(f"  Train windows: {len(X_train):,} ({y_train.mean():.1%} hard)")
    print(f"  Val windows:   {len(X_val):,} ({y_val.mean():.1%} hard)")

    if len(X_val) < 10:
        print("  ERROR: Too few validation windows. Skipping.")
        return None, None

    X_train_rf = build_rf_features(X_train)
    X_val_rf = build_rf_features(X_val)

    # ── Train ──
    print(f"\n  Training LSTM for {target_label} ...")
    lstm, history = train_lstm(X_train, y_train, X_val, y_val,
                               len(SELECTED_FEATURES), device)

    print(f"\n  Training RF for {target_label} ...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                min_samples_leaf=10, max_features="sqrt",
                                n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train)

    # ── Predictions ──
    with torch.no_grad():
        lstm_pred, lstm_prob, lstm_unc = lstm.predict(
            torch.from_numpy(X_val).to(device))
    lstm_pred = lstm_pred.cpu().numpy()
    lstm_prob = lstm_prob.cpu().numpy()
    lstm_unc = lstm_unc.cpu().numpy()

    rf_pred = rf.predict(X_val_rf)
    rf_prob = rf.predict_proba(X_val_rf)[:, 1]

    # ── Baselines ──
    majority_class = int(y_train.mean() >= 0.5)
    majority_pred = np.full_like(y_val, majority_class)

    fast_idx = SELECTED_FEATURES.index("fast_keypoints_half")
    train_fast = X_train[:, :, fast_idx].mean(axis=1)
    val_fast = X_val[:, :, fast_idx].mean(axis=1)
    best_fast_acc, best_fast_thr = 0, 0
    for q in np.arange(0.05, 0.95, 0.02):
        thr = np.quantile(train_fast, q)
        preds = (train_fast >= thr).astype(int)
        acc = accuracy_score(y_train, preds)
        if acc > best_fast_acc:
            best_fast_acc = acc
            best_fast_thr = thr
    fast_pred = (val_fast >= best_fast_thr).astype(int)

    # Persistence (use last training label for first prediction, not val[0])
    persist_pred = np.empty_like(y_val)
    persist_pred[0] = y_train[-1]
    persist_pred[1:] = y_val[:-1]

    # ── MedSample baseline (fn_nano only) ──
    medsample_pred = None
    if target_col == "miss_rate_nano":
        val_sorted = val_df.sort_values("frame_id").reset_index(drop=True)
        frame_to_idx = {fid: i for i, fid in enumerate(val_sorted["frame_id"])}
        medsample_pred = np.zeros(len(val_ids), dtype=np.int64)
        for w, (seq, start_fid) in enumerate(val_ids):
            start_idx = frame_to_idx[start_fid]
            last_idx = start_idx + WINDOW - 1
            med_tp = val_sorted["medium_tp"].iloc[last_idx]
            nan_tp = val_sorted["nano_tp"].iloc[last_idx]
            medsample_pred[w] = int(med_tp > nan_tp)

    # ── Evaluate ──
    print(f"\n  EVALUATION — {target_label}")
    print("  " + "=" * 60)

    results = {}
    results["LSTM"] = eval_clf(y_val, lstm_pred, lstm_prob, "LSTM")
    results["RF Classifier"] = eval_clf(y_val, rf_pred, rf_prob, "RF Classifier")
    results["FAST Thr"] = eval_clf(y_val, fast_pred, None, "FAST Threshold")
    results["FAST Thr"]["auc"] = 0.0
    results["Persistence"] = eval_clf(y_val, persist_pred, None, "Persistence")
    results["Persistence"]["auc"] = 0.0
    results["Majority Class"] = eval_clf(y_val, majority_pred, None, "Majority Class")
    results["Majority Class"]["auc"] = 0.0
    if medsample_pred is not None:
        results["MedSample"] = eval_clf(y_val, medsample_pred, None, "MedSample")
        results["MedSample"]["auc"] = 0.0

    # Transition accuracy
    print(f"\n  TRANSITION ACCURACY — {target_label}")
    trans_pairs = [("LSTM", lstm_pred), ("RF Classifier", rf_pred),
                    ("FAST Thr", fast_pred), ("Persistence", persist_pred),
                    ("Majority Class", majority_pred)]
    if medsample_pred is not None:
        trans_pairs.append(("MedSample", medsample_pred))
    for name, preds in trans_pairs:
        ta, n_trans = transition_accuracy(y_val, preds)
        if name in results:
            results[name]["trans_acc"] = ta
            results[name]["n_trans"] = n_trans
        print(f"    {name:25s}: {ta:.3f} ({n_trans} transitions)")

    # Abstention
    lstm_correct = (lstm_pred == y_val)
    abstention_results = []
    for unc_pct in [0, 10, 20, 30, 40, 50]:
        if unc_pct == 0:
            mask = np.ones(len(y_val), dtype=bool)
        else:
            unc_threshold = np.percentile(lstm_unc, 100 - unc_pct)
            mask = lstm_unc <= unc_threshold
        n_kept = mask.sum()
        if n_kept == 0:
            continue
        acc_kept = accuracy_score(y_val[mask], lstm_pred[mask])
        combined_pred = lstm_pred.copy()
        combined_pred[~mask] = 1
        acc_combined = accuracy_score(y_val, combined_pred)
        hard_mask = y_val == 1
        recall_combined = (combined_pred[hard_mask] == 1).mean() if hard_mask.any() else 0.0
        abstention_results.append({
            "abstain_pct": unc_pct, "n_kept": n_kept,
            "acc_confident": acc_kept, "acc_combined": acc_combined,
            "recall_hard": recall_combined,
        })

    artifacts = {
        "target_col": target_col,
        "target_label": target_label,
        "threshold": threshold,
        "y_val": y_val, "val_ids": val_ids,
        "lstm_pred": lstm_pred, "lstm_prob": lstm_prob, "lstm_unc": lstm_unc,
        "lstm_correct": lstm_correct,
        "rf_pred": rf_pred, "rf_prob": rf_prob, "rf": rf,
        "fast_pred": fast_pred, "persist_pred": persist_pred,
        "majority_pred": majority_pred,
        "medsample_pred": medsample_pred,
        "history": history,
        "abstention_results": abstention_results,
    }
    return results, artifacts


def evaluate_and_report(features_df):
    """Train models for both miss_rate_nano and miss_rate_medium, evaluate, generate PDF."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Compute miss rates: fn / x_count (safe division)
    features_df["miss_rate_nano"] = np.where(
        features_df["x_count"] > 0,
        features_df["fn_nano"] / features_df["x_count"], 0.0)
    features_df["miss_rate_medium"] = np.where(
        features_df["x_count"] > 0,
        features_df["fn_medium"] / features_df["x_count"], 0.0)

    targets = [
        ("miss_rate_nano", "YOLO11n@320 miss rate"),
        ("miss_rate_medium", "YOLO11m@640 miss rate"),
    ]

    all_results = {}  # {target_col: results_dict}
    all_artifacts = {}

    for target_col, target_label in targets:
        results, artifacts = run_single_target(
            features_df, target_col, target_label, device)
        if results is not None:
            all_results[target_col] = results
            all_artifacts[target_col] = artifacts

    # ═══════════════════════════════════════════════════════════════
    # PER-SEQUENCE NORM
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PER-SEQUENCE NORMALIZATION")
    print("=" * 65)
    print("  N/A: Single sequence ('fresno'). Per-seq norm would zero out")
    print("  all feature variance, leaving only noise. This is by design.")

    # ═══════════════════════════════════════════════════════════════
    # SAVE RESULTS CSV (both targets)
    # ═══════════════════════════════════════════════════════════════
    csv_rows = []
    for target_col, results in all_results.items():
        for name, m in results.items():
            csv_rows.append({
                "target": target_col,
                "model": name, "acc": m["acc"], "f1": m["f1"],
                "prec": m["prec"], "rec": m["rec"], "auc": m["auc"],
                "trans_acc": m.get("trans_acc", None),
                "n_transitions": m.get("n_trans", None),
            })
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    print(f"\n  Results CSV: {OUT_CSV}")

    # ═══════════════════════════════════════════════════════════════
    # PDF REPORT
    # ═══════════════════════════════════════════════════════════════
    print("  Generating PDF ...", flush=True)
    colors = ["#1976D2", "#388E3C", "#7B1FA2", "#F57C00", "#455A64", "#C62828"]

    with PdfPages(OUT_PDF) as pdf:
        for target_col in all_results:
            results = all_results[target_col]
            art = all_artifacts[target_col]
            tl = art["target_label"]
            threshold = art["threshold"]
            y_val = art["y_val"]
            val_ids = art["val_ids"]
            lstm_pred = art["lstm_pred"]
            lstm_unc = art["lstm_unc"]
            lstm_correct = art["lstm_correct"]
            rf_pred = art["rf_pred"]
            persist_pred = art["persist_pred"]
            rf = art["rf"]
            history = art["history"]
            abstention_results = art["abstention_results"]
            model_names = list(results.keys())

            # ── Page A: Accuracy/F1 bar chart ──
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            for ax_idx, metric in enumerate(["acc", "f1"]):
                ax = axes[ax_idx]
                vals = [results[m][metric] for m in model_names]
                x = np.arange(len(model_names))
                bars = ax.bar(x, vals, color=colors[:len(model_names)])
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
                ax.set_ylabel(metric.upper())
                ax.set_title(f"{'Accuracy' if metric == 'acc' else 'F1 Score'}")
                ax.axhline(max(y_val.mean(), 1 - y_val.mean()), color="gray",
                           ls=":", lw=1, label="Majority baseline")
                ax.legend(fontsize=8)
                ax.set_ylim(0.0, 1.0)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                            f"{v:.3f}", ha="center", fontsize=8)
            fig.suptitle(f"Fresno: {tl}\n"
                         f"Threshold: {target_col} >= {threshold:.1f}",
                         fontsize=12, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig)
            plt.close()

            # ── Page B: Confusion matrices ──
            medsample_pred = art.get("medsample_pred", None)
            cm_panels = [
                ("LSTM", lstm_pred),
                ("RF Classifier", rf_pred),
                ("Persistence", persist_pred),
            ]
            if medsample_pred is not None:
                cm_panels.append(("MedSample", medsample_pred))
            n_panels = len(cm_panels)
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 5))
            if n_panels == 1:
                axes = [axes]
            for idx, (name, preds) in enumerate(cm_panels):
                ax = axes[idx]
                cm = confusion_matrix(y_val, preds)
                ax.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        color = "white" if cm[i, j] > cm.max() / 2 else "black"
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                fontsize=16, color=color)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Pred Easy", "Pred Hard"])
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["True Easy", "True Hard"])
                r = results.get(name, {})
                ax.set_title(f"{name}\nAcc={r.get('acc', 0):.3f} "
                             f"Trans={r.get('trans_acc', 0):.3f}", fontsize=10)
            fig.suptitle(f"Confusion Matrices — {tl}", fontsize=13, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close()

            # ── Page C: Temporal plot ──
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))
            ax = axes[0]
            frame_ids = features_df["frame_id"].values
            fn_vals = features_df[target_col].values
            ax.plot(frame_ids, fn_vals, linewidth=0.3, alpha=0.6, color="steelblue")
            win = min(60, len(fn_vals) // 10)
            if win > 1:
                smoothed = pd.Series(fn_vals).rolling(win, center=True, min_periods=1).mean()
                ax.plot(frame_ids, smoothed, linewidth=1.5, color="darkblue",
                        label=f"Smoothed (window={win})")
            ax.axvline(TRAIN_CUTOFF, color="red", ls="--", lw=2, label="Train/Val split")
            ax.set_xlabel("Frame ID")
            ax.set_ylabel(f"{target_col}")
            ax.set_title(f"{target_col} Over Time")
            ax.legend(fontsize=9)

            ax = axes[1]
            val_frame_ids = [fid for _, fid in val_ids]
            ax.plot(val_frame_ids, y_val, "k-", alpha=0.4, linewidth=0.8, label="True label")
            ax.plot(val_frame_ids, lstm_pred, "b.", markersize=3, alpha=0.5, label="LSTM pred")
            ax.plot(val_frame_ids, rf_pred + 0.05, "g.", markersize=3, alpha=0.5, label="RF pred")
            ax.set_xlabel("Frame ID (val windows)")
            ax.set_ylabel("Label (0=easy, 1=hard)")
            ax.set_title(f"Predictions vs Ground Truth — {tl}")
            ax.legend(fontsize=9)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # ── Page D: Uncertainty + training curves ──
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            ax = axes[0, 0]
            if lstm_correct.any():
                ax.hist(lstm_unc[lstm_correct], bins=20, alpha=0.6,
                        label=f"Correct ({lstm_unc[lstm_correct].mean():.3f})",
                        color="green", density=True)
            if (~lstm_correct).any():
                ax.hist(lstm_unc[~lstm_correct], bins=20, alpha=0.6,
                        label=f"Incorrect ({lstm_unc[~lstm_correct].mean():.3f})",
                        color="red", density=True)
            ax.set_xlabel("Uncertainty (K/S)")
            ax.set_title(f"Uncertainty — {tl}")
            ax.legend(fontsize=9)

            ax = axes[0, 1]
            if abstention_results:
                abs_pcts = [r["abstain_pct"] for r in abstention_results]
                ax.plot(abs_pcts, [r["acc_confident"] for r in abstention_results],
                        "bo-", label="Acc (confident)")
                ax.plot(abs_pcts, [r["acc_combined"] for r in abstention_results],
                        "gs-", label="Acc (combined)")
                ax.plot(abs_pcts, [r["recall_hard"] for r in abstention_results],
                        "r^-", label="Recall (hard)")
                ax.legend(fontsize=8)
            ax.set_xlabel("% Abstained")
            ax.set_title("Abstention Curve")
            ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            ax.plot(history["train_acc"], "b-", alpha=0.7, label="Train")
            ax.plot(history["val_acc"], "r-", alpha=0.7, label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("LSTM Training Curves")
            ax.legend()
            ax.set_ylim(0.4, 1.0)

            ax = axes[1, 1]
            trans_names = [n for n in model_names if "trans_acc" in results.get(n, {})]
            trans_vals = [results[n]["trans_acc"] for n in trans_names]
            x = np.arange(len(trans_names))
            bars = ax.bar(x, trans_vals, color=colors[:len(trans_names)])
            ax.set_xticks(x)
            ax.set_xticklabels(trans_names, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Transition Accuracy")
            ax.set_title(f"Transition Accuracy — {tl}")
            ax.axhline(0.5, color="red", ls="--", lw=1, label="Random chance")
            ax.legend(fontsize=8)
            ax.set_ylim(0.0, 1.0)
            for bar, v in zip(bars, trans_vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.3f}", ha="center", fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # ── Page E: RF feature importances ──
            fig, ax = plt.subplots(figsize=(12, 7))
            feat_imp = rf.feature_importances_
            labels = ([f"{f}_mean" for f in SELECTED_FEATURES] +
                      [f"{f}_std" for f in SELECTED_FEATURES] +
                      [f"{f}_slope" for f in SELECTED_FEATURES])
            sorted_idx = np.argsort(feat_imp)[-20:]
            ax.barh(range(20), feat_imp[sorted_idx], color="steelblue")
            ax.set_yticks(range(20))
            ax.set_yticklabels([labels[i] for i in sorted_idx], fontsize=8)
            ax.set_xlabel("Feature Importance (RF)")
            ax.set_title(f"Top 20 RF Features — {tl}")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # ── Final Page: Combined summary table ──
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.axis("off")

        header = ["Target", "Model", "Acc", "F1", "Prec", "Rec", "AUC",
                  "Trans Acc", "Transitions"]
        rows = []
        for target_col, results in all_results.items():
            for name, m in results.items():
                rows.append([
                    target_col, name, f"{m['acc']:.3f}", f"{m['f1']:.3f}",
                    f"{m['prec']:.3f}", f"{m['rec']:.3f}",
                    f"{m['auc']:.3f}" if m['auc'] > 0 else "---",
                    f"{m.get('trans_acc', 0):.3f}",
                    str(m.get("n_trans", "---")),
                ])

        table = ax.table(cellText=rows, colLabels=header,
                         loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.6)
        for j in range(len(header)):
            table[0, j].set_facecolor("#37474F")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Shade fn_medium rows differently
        for i, row_data in enumerate(rows):
            if row_data[0] == "fn_medium":
                for j in range(len(header)):
                    table[i + 1, j].set_facecolor("#E3F2FD")

        summary = (
            f"Fresno Traffic Camera — Nano + Medium vs X (Pseudo GT)\n"
            f"{'=' * 70}\n"
            f"Video: 3h OBS recording, 960x540 at 2fps ({len(features_df):,} frames)\n"
            f"Temporal split: Train first 2h | Val last 1h\n"
            f"Per-sequence normalization: N/A (single sequence)\n"
        )
        ax.text(0.5, 0.15, summary, transform=ax.transAxes,
                ha="center", va="top", fontfamily="monospace", fontsize=9)
        ax.set_title("Fresno Experiment — Combined Results (miss_rate_nano + miss_rate_medium)",
                     fontsize=14, fontweight="bold", pad=40)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"  PDF: {OUT_PDF}")
    print("\nDone.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FRESNO TRAFFIC CAMERA EXPERIMENT")
    print("YOLO11n@320 vs YOLO11x@640 — Temporal Split Pipeline")
    print("=" * 70)
    print(f"  Video:   {VIDEO}")
    print(f"  Output:  {SCRIPT_DIR}")
    print()

    # Phase 1
    n_frames = extract_frames()

    # Phase 2
    all_dets = run_yolo_inference(n_frames)

    # Phase 3
    det_df = compute_matching(all_dets)

    # Phase 4
    features_df = run_feature_extraction(det_df)

    # Phase 5-7
    print("\n" + "=" * 70)
    print("PHASE 5-7: WINDOWING, TRAINING, EVALUATION")
    print("=" * 70)
    evaluate_and_report(features_df)


if __name__ == "__main__":
    main()
