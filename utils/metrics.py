import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """Computes basic metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)

    print(f"--- {model_name} Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if auc:
        print(f"ROC-AUC:   {auc:.4f}")
    print("-" * 30)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def plot_roc_curves(results_dict, y_true_dict, save_dir="results"):
    """Plots ROC curves for multiple models."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    for model_name, y_prob in results_dict.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true_dict[model_name], y_prob)
            auc = roc_auc_score(y_true_dict[model_name], y_prob)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    print(f"ROC curves saved to {os.path.join(save_dir, 'roc_curves.png')}")
    plt.close()
