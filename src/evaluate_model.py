"""
evaluate_model.py
------------------
Evaluates the trained model using Accuracy, Precision, Recall, F1-score,
Confusion Matrix, and ROC-AUC. Saves evaluation plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)


def evaluate(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm, y_pred, y_proba


def plot_confusion_matrix(cm, target_names, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                cbar=False, annot_kws={"size": 14})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_proba, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#2980b9", linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_coefficients(model, feature_names, save_path, top_n=15):
    """Logistic regression coefficients as a simple, fast feature-importance view."""
    import numpy as np
    import pandas as pd
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    top = pd.concat([coefs.head(top_n // 2), coefs.tail(top_n // 2)])
    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in top.values]
    plt.barh(top.index, top.values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Logistic Regression Coefficients\n(negative = pushes toward Malignant, positive = pushes toward Benign)")
    plt.xlabel("Coefficient value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
