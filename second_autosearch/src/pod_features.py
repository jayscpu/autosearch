# Single source of truth for the feature lists.
# Import this in every training/evaluation script.

# Original 65 spatial features (from Fresno experiment)
SPATIAL_65 = [
    "fft_high_freq_energy_ratio", "fft_critical_band_ratio",
    "laplacian_variance", "laplacian_mean_abs",
    "edge_density_fine", "edge_density_coarse", "edge_fine_coarse_ratio",
    "fast_keypoints_full", "fast_keypoints_half", "keypoint_loss_ratio",
    "spatial_frequency", "gradient_magnitude_mean", "gradient_magnitude_std",
    "dct_high_freq_ratio", "gabor_nyquist_energy",
    "downsample_ssim", "downsample_info_loss",
    "foreground_pixel_ratio", "shadow_pixel_ratio",
    "foreground_blob_count", "foreground_blob_area_mean",
    "foreground_blob_area_std", "foreground_blob_pa_ratio",
    "foreground_edge_density",
    "temporal_diff_mean", "temporal_diff_std", "motion_pixel_ratio",
    "image_entropy",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_entropy",
    "colorfulness", "mean_brightness", "brightness_std",
    "dark_channel_mean",
    "small_edge_components", "total_edge_components",
    "overall_edge_density", "rms_contrast",
    "saturation_mean", "saturation_std",
    "mscn_mean", "mscn_std", "mscn_kurtosis", "mscn_skewness",
    "mscn_h_pair_mean", "mscn_h_pair_std", "mscn_v_pair_mean", "mscn_v_pair_std",
    "top_laplacian_var", "top_gradient_std", "top_edge_density", "top_high_freq_energy",
    "mid_laplacian_var", "mid_gradient_std", "mid_edge_density", "mid_high_freq_energy",
    "bot_laplacian_var", "bot_gradient_std", "bot_edge_density", "bot_high_freq_energy",
    "ratio_top_bot_edge_density", "ratio_top_bot_gradient_std", "ratio_top_bot_laplacian_var",
]

# New temporal/dynamic features (added for Bellevue)
NEW_FEATURES = [
    # Optical flow
    "optical_flow_magnitude_mean", "optical_flow_magnitude_std",
    "optical_flow_direction_entropy",
    # SSIM scene change
    "ssim_vs_prev",  # SSIM between current and previous frame
    # Detector-derived (from YOLO11n — "free" since we run it anyway)
    "det_confidence_mean",       # mean detection confidence (nano)
    "det_spatial_dispersion",    # std of bbox centroids (nano)
    "det_small_object_ratio",    # proportion of small detections (nano)
    "det_count_nano",            # raw detection count from nano
    # Detector-derived (from YOLO11s)
    "det_count_small",           # raw detection count from small
    "det_confidence_small",      # mean detection confidence (small)
]

# Combined full feature set
ALL_FEATURES = SPATIAL_65 + NEW_FEATURES

# Delta (rate-of-change) features — computed from CSV before windowing
DELTA_BASE_FEATURES = [
    "foreground_blob_count", "foreground_pixel_ratio", "motion_pixel_ratio",
    "temporal_diff_mean", "optical_flow_magnitude_mean", "edge_density_coarse",
    "image_entropy", "mean_brightness",
]
DELTA_FEATURES = [f"delta5_{f}" for f in DELTA_BASE_FEATURES] + \
                 [f"delta10_{f}" for f in DELTA_BASE_FEATURES]

# Top-35 Spearman features from Fresno (kept for backward compatibility)
TOP_35_SPEARMAN = [
    "glcm_entropy", "image_entropy", "glcm_energy", "brightness_std",
    "rms_contrast", "glcm_contrast", "mscn_v_pair_mean",
    "gabor_nyquist_energy", "foreground_edge_density", "edge_density_coarse",
    "foreground_pixel_ratio", "edge_fine_coarse_ratio", "fast_keypoints_half",
    "shadow_pixel_ratio", "glcm_homogeneity", "gradient_magnitude_std",
    "spatial_frequency", "foreground_blob_count", "mid_gradient_std",
    "downsample_info_loss", "downsample_ssim", "mscn_mean",
    "motion_pixel_ratio", "mid_high_freq_energy", "fft_critical_band_ratio",
    "temporal_diff_mean", "mscn_skewness", "temporal_diff_std",
    "keypoint_loss_ratio", "saturation_std", "ratio_top_bot_gradient_std",
    "bot_gradient_std", "dark_channel_mean", "colorfulness", "mscn_h_pair_std",
]
