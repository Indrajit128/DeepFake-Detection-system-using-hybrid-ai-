import os

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
REAL_DIR    = os.path.join(DATA_DIR, "real")
FAKE_DIR    = os.path.join(DATA_DIR, "fake")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "saved_models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Image settings -- 128x128 keeps VRAM well under 2 GB
IMG_SIZE    = 128
CHANNELS    = 3

# Training
BATCH_SIZE  = 32
EPOCHS_CNN  = 30
EPOCHS_CBAM = 30
LEARNING_RATE = 1e-4

# Data split
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15   # remainder becomes test

# CBAM
CBAM_REDUCTION_RATIO = 8

# ML feature extraction
PCA_COMPONENTS = 128   # PCA dims fed to SVM/Ridge/Lasso

# Reproducibility
SEED = 42
