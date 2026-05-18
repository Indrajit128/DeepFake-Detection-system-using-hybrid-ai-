"""
Evaluation utilities: metrics computation, ROC/AUC curves, confusion matrix,
and a side-by-side comparison table for all models.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless -- no display required

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
import config


os.makedirs(config.PLOTS_DIR,   exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """
    y_pred : binary predictions (0/1)
    y_prob : predicted probability for the positive class (fake)
    """
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)),  4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred,    zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred,        zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def print_metrics(name: str, metrics: dict):
    bar = "-" * 40
    print(f"\n{bar}")
    print(f"  {name}")
    print(bar)
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")
    print(bar)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curve(roc_data: dict, save_path: str = None):
    """
    roc_data: {model_name: (y_true, y_prob)}
    """
    plt.figure(figsize=(7, 6))
    for name, (y_true, y_prob) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves -- Deepfake Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = save_path or os.path.join(config.PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curves saved -> {path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"cm_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {path}")


def plot_training_history(history, model_name: str):
    """Plot accuracy and loss curves from a Keras History object."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"{model_name} -- Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"{model_name} -- Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f"history_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training history saved -> {path}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(all_metrics: dict) -> str:
    """
    all_metrics: {model_name: metrics_dict}
    Returns a pretty-printed table string and saves it as CSV + JSON.
    """
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    sep    = "-" * len(header)
    rows   = [sep, header, sep]

    for name, m in all_metrics.items():
        row = (
            f"{name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}"
        )
        rows.append(row)
    rows.append(sep)
    table = "\n".join(rows)
    print("\n" + table)

    # Save CSV
    csv_path = os.path.join(config.RESULTS_DIR, "comparison.csv")
    with open(csv_path, "w") as f:
        f.write("model,accuracy,precision,recall,f1,roc_auc\n")
        for name, m in all_metrics.items():
            f.write(f"{name},{m['accuracy']},{m['precision']},"
                    f"{m['recall']},{m['f1']},{m['roc_auc']}\n")

    # Save JSON
    json_path = os.path.join(config.RESULTS_DIR, "comparison.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved -> {csv_path}\n         -> {json_path}")
    return table


def plot_metric_bars(all_metrics: dict):
    """Grouped bar chart comparing all models across all metrics."""
    model_names = list(all_metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(model_names):
        vals = [all_metrics[name][k] for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=name)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- Deepfake Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "metric_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Bar comparison saved -> {path}")
