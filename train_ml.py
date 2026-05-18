"""
Train and evaluate SVM, Ridge, and Lasso classifiers on HOG+PCA features.
Results are saved to outputs/results/ and plots to outputs/plots/.

Usage:
    python train_ml.py
"""

import numpy as np
from utils.dataset import load_dataset, split_dataset
from utils.feature_extractor import prepare_ml_features
from utils.evaluation import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, build_comparison_table, plot_metric_bars,
)
from models.ml_models import ALL_ML_MODELS, get_predict_proba, save_ml_model
import config


def main():
    # 1. Data
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # 2. Features
    F_train, F_val, F_test, _ = prepare_ml_features(X_train, X_val, X_test)

    # Combine train+val for final model training
    F_trainval = np.concatenate([F_train, F_val], axis=0)
    y_trainval  = np.concatenate([y_train, y_val], axis=0)

    all_metrics = {}
    roc_data    = {}

    for name, builder in ALL_ML_MODELS.items():
        print(f"\n{'='*50}\nTraining {name}...\n{'='*50}")
        model = builder()
        model.fit(F_trainval, y_trainval)
        save_ml_model(model, name)

        y_pred = model.predict(F_test)
        y_prob = get_predict_proba(model, F_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(name, metrics)
        all_metrics[name] = metrics
        roc_data[name]    = (y_test, y_prob)

        plot_confusion_matrix(y_test, y_pred, name)

    print("\n\n===== COMPARISON =====")
    build_comparison_table(all_metrics)
    plot_metric_bars(all_metrics)

    # ROC — defer to evaluate.py which combines all models
    return all_metrics, roc_data


if __name__ == "__main__":
    main()
