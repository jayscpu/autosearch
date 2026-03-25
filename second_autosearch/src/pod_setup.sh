#!/bin/bash
# pod_setup.sh — Run this first on the RunPod instance.
# Installs dependencies, downloads dataset, clones repo.
set -e

echo "=== ECHO Bellevue Pod Setup ==="
cd /workspace

# ── Dependencies ──
# Don't install torch/torchvision — RunPod template has CUDA-matched versions.
# pip installing them would overwrite with CPU-only builds.
pip install -q ultralytics opencv-python-headless gdown pandas numpy \
    scikit-learn xgboost scipy

# ── Clone repo ──
if [ ! -d "autosearch" ]; then
    git clone https://github.com/jayscpu/autosearch.git
fi

# ── Create working directory ──
mkdir -p autosearch/pod/data
mkdir -p autosearch/pod/frames
mkdir -p autosearch/pod/checkpoints

echo ""
echo "=== Setup complete. Now run: bash pod_download.sh ==="
