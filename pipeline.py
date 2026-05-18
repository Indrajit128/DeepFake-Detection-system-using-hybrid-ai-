"""
Full end-to-end pipeline: trains all models sequentially and runs final comparison.

Usage:
    python pipeline.py
"""

import sys
import train_ml
import train_cnn
import train_cbam
from utils.evaluation import build_comparison_table, plot_roc_curve, plot_metric_bars
import config


def main():
    print("\n" + "="*60)
    print("  DEEPFAKE DETECTION — FULL PIPELINE")
    print("="*60)

    all_metrics = {}
    roc_data    = {}

    # ── 1. Classical ML ────────────────────────────────────────────────────
    print("\n[1/3] Training classical ML models (SVM, Ridge, Lasso)...")
    ml_metrics, ml_roc = train_ml.main()
    all_metrics.update(ml_metrics)
    roc_data.update(ml_roc)

    # ── 2. CNN ─────────────────────────────────────────────────────────────
    print("\n[2/3] Training baseline CNN...")
    cnn_metrics, y_test, cnn_prob = train_cnn.main()
    all_metrics["CNN"]  = cnn_metrics
    roc_data["CNN"]     = (y_test, cnn_prob)

    # ── 3. CNN + CBAM ──────────────────────────────────────────────────────
    print("\n[3/3] Training CNN + CBAM...")
    cbam_metrics, y_test, cbam_prob = train_cbam.main()
    all_metrics["CNN+CBAM"] = cbam_metrics
    roc_data["CNN+CBAM"]    = (y_test, cbam_prob)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    build_comparison_table(all_metrics)
    plot_metric_bars(all_metrics)
    plot_roc_curve(roc_data)

    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
