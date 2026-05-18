"""
Classical ML classifiers: SVM (RBF), Ridge, and Lasso (L1 logistic regression).
Provides factory functions and shared utilities used by train_ml.py.
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import config


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_svm_model() -> CalibratedClassifierCV:
    """SVM with RBF kernel, Platt scaling for probability estimates."""
    base = SVC(kernel="rbf", probability=True,
               gamma="scale", random_state=config.SEED)
    return CalibratedClassifierCV(base, cv=3)


def get_ridge_model() -> RidgeClassifier:
    """Ridge Classifier (L2 regularised linear model)."""
    return RidgeClassifier(class_weight="balanced")


def get_lasso_model() -> LogisticRegression:
    """Logistic Regression with L1 penalty (Lasso equivalent for classification)."""
    return LogisticRegression(
        penalty="l1", solver="liblinear",
        C=1.0, class_weight="balanced",
        max_iter=2000, random_state=config.SEED,
    )


ALL_ML_MODELS = {
    "SVM":   get_svm_model,
    "Ridge": get_ridge_model,
    "Lasso": get_lasso_model,
}


# ---------------------------------------------------------------------------
# Unified probability getter (handles models without predict_proba)
# ---------------------------------------------------------------------------

def get_predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Returns P(fake) for each sample, regardless of model type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # RidgeClassifier → decision_function, normalise to [0, 1]
    scores = model.decision_function(X)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-9:
        return np.full(len(scores), 0.5)
    return (scores - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_ml_model(model, name: str):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Saved {name} → {path}")


def load_ml_model(name: str):
    return joblib.load(os.path.join(config.MODELS_DIR, f"{name}.pkl"))
