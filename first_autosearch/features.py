# Single source of truth for the top-35 feature list (by Spearman correlation).
# Import this in every training/evaluation script instead of duplicating.

TOP_35_FEATURES = [
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
