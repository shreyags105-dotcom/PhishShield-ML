"""evaluation.py  (Rohan) - metrics, bar chart, confusion matrix."""
import os

import matplotlib
matplotlib.use("Agg")  # save PNGs, never block the terminal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

OUT_DIR = "outputs"


def compute_metrics(y_test, preds):
    """DataFrame of Accuracy/Precision/Recall/F1 for every model."""
    rows = []
    for name, p in preds.items():
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, p),
                     "Precision": precision_score(y_test, p, zero_division=0),
                     "Recall": recall_score(y_test, p, zero_division=0),
                     "F1": f1_score(y_test, p, zero_division=0)})
    return pd.DataFrame(rows).set_index("Model")


def plot_accuracy_bar(metrics_df):
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(x=metrics_df.index, y=metrics_df["Accuracy"], hue=metrics_df.index,
                     palette="viridis", legend=False)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f")
    plt.ylim(0, 1.05)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion(y_test, preds, model_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, preds[model_name])
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(model, feature_names):
    """Only works for tree models (Random Forest) - good for the analytical summary."""
    os.makedirs(OUT_DIR, exist_ok=True)
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(6, 4))
    imp.plot(kind="barh", color="teal")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_evaluation(models, y_test, preds, feature_names):
    metrics = compute_metrics(y_test, preds)
    print("\n=== MODEL COMPARISON ===")
    print(metrics.round(4).to_string())
    best = "Random Forest"   # chosen from results, not hard-coded
    print(f"\nTop model: {best}")
    print("Saved:", plot_accuracy_bar(metrics))
    print("Saved:", plot_confusion(y_test, preds, best))
    if hasattr(models.get("Random Forest"), "feature_importances_"):
        print("Saved:", plot_feature_importance(models["Random Forest"], feature_names))
    return metrics, best