"""
main.py
--------
End-to-end pipeline: data loading -> preprocessing -> EDA -> training ->
evaluation -> SHAP explainability -> LIME explainability.

Run with:  python main.py
(from inside the src/ directory)
"""

import json
import joblib

from data_preprocessing import load_data, split_and_scale
from eda import plot_class_distribution, plot_correlation_heatmap
from train_model import train_logistic_regression
from evaluate_model import (
    evaluate, plot_confusion_matrix, plot_roc_curve, plot_coefficients
)

FIG_DIR = "../outputs/figures"
MODEL_DIR = "../models"


def main():
    print("1. Loading data...")
    X, y, feature_names, target_names = load_data()
    print(f"   -> {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   -> Classes: {target_names}")

    print("2. EDA...")
    plot_class_distribution(y, target_names, f"{FIG_DIR}/class_distribution.png")
    plot_correlation_heatmap(X, f"{FIG_DIR}/correlation_heatmap.png")

    print("3. Preprocessing (train/test split + scaling)...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print("4. Training Logistic Regression...")
    model = train_logistic_regression(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/logistic_regression_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("5. Evaluating...")
    metrics, report, cm, y_pred, y_proba = evaluate(model, X_test, y_test, target_names)
    plot_confusion_matrix(cm, target_names, f"{FIG_DIR}/confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, f"{FIG_DIR}/roc_curve.png")
    plot_coefficients(model, feature_names, f"{FIG_DIR}/logistic_regression_coefficients.png")

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}  ({v*100:.2f}%)")
    print("\n=== Classification Report ===")
    print(report)

    with open(f"{FIG_DIR}/../metrics.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=2)

    print("\n6. SHAP + LIME explainability...")
    try:
        from explainability import run_shap_analysis, run_lime_analysis
        run_shap_analysis(model, X_train, X_test, feature_names, target_names)
        run_lime_analysis(model, X_train, X_test, feature_names, target_names, sample_index=0)
    except ImportError as e:
        print(f"   -> Skipped (install shap & lime to run this step): {e}")

    print("\nDone. Figures saved in outputs/figures/, model saved in models/.")


if __name__ == "__main__":
    main()
