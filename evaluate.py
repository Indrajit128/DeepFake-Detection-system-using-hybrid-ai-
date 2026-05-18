"""
Unified evaluation: runs all five models on the test set and produces
a combined ROC curve + bar chart comparison.

Usage:
    python evaluate.py
    (assumes models have already been trained via train_ml.py / train_cnn.py / train_cbam.py)
"""

import os
import numpy as np
import tensorflow as tf

from utils.dataset import load_dataset, split_dataset
from utils.feature_extractor import prepare_ml_features, FeaturePipeline
from utils.evaluation import (
    compute_metrics, print_metrics,
    plot_roc_curve, plot_confusion_matrix,
    build_comparison_table, plot_metric_bars,
)
from models.ml_models import ALL_ML_MODELS, get_predict_proba, load_ml_model
import config


def main():
    # ── Data ───────────────────────────────────────────────────────────────
    X, y = load_dataset()
    _, _, X_test, _, _, y_test = split_dataset(X, y)

    # ── ML features ────────────────────────────────────────────────────────
    from skimage.feature import hog

    def extract_hog(images):
        feats = []
        for img in images:
            img_u = (img * 255).astype("uint8")
            h = hog(img_u, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), channel_axis=-1, feature_vector=True)
            feats.append(h)
        return np.array(feats, dtype=np.float32)

    pipe = FeaturePipeline()
    pipe.load("hog")
    F_test = pipe.transform(extract_hog(X_test))

    all_metrics = {}
    roc_data    = {}

    # ── ML models ──────────────────────────────────────────────────────────
    for name in ALL_ML_MODELS:
        pkl = os.path.join(config.MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(pkl):
            print(f"[SKIP] {name} — model not found ({pkl})")
            continue
        model  = load_ml_model(name)
        y_pred = model.predict(F_test)
        y_prob = get_predict_proba(model, F_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(name, m)
        all_metrics[name]  = m
        roc_data[name]     = (y_test, y_prob)
        plot_confusion_matrix(y_test, y_pred, name)

    # ── CNN ────────────────────────────────────────────────────────────────
    for model_tag, display_name in [("cnn_best", "CNN"), ("cbam_best", "CNN+CBAM")]:
        ckpt = os.path.join(config.MODELS_DIR, f"{model_tag}.keras")
        if not os.path.exists(ckpt):
            print(f"[SKIP] {display_name} — model not found ({ckpt})")
            continue
        model  = tf.keras.models.load_model(ckpt)
        y_prob = model.predict(X_test, batch_size=config.BATCH_SIZE).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        m = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(display_name, m)
        all_metrics[display_name] = m
        roc_data[display_name]    = (y_test, y_prob)
        plot_confusion_matrix(y_test, y_pred, display_name)

    # ── Summary ────────────────────────────────────────────────────────────
    if not all_metrics:
        print("No trained models found. Run train_ml.py, train_cnn.py, train_cbam.py first.")
        return

    print("\n\n===== FULL COMPARISON =====")
    build_comparison_table(all_metrics)
    plot_metric_bars(all_metrics)
    plot_roc_curve(roc_data)


if __name__ == "__main__":
    main()
