"""
explainability.py
-------------------
Model interpretability using SHAP (global + local explanations) and
LIME (local explanations for individual predictions).

Requires: pip install shap lime
(These are pre-installed in a Kaggle notebook environment.)
"""

import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer


# ----------------------------- SHAP -----------------------------------

def run_shap_analysis(model, X_train, X_test, feature_names, target_names,
                       save_dir="../outputs/figures"):
    """
    Uses SHAP's LinearExplainer (fast + exact for linear models like
    Logistic Regression) to compute Shapley values, then produces:
      1. A global summary (beeswarm) plot -> which features matter most overall
      2. A bar plot of mean |SHAP value| per feature -> global feature importance
      3. A force/waterfall plot for a single test example -> local explanation
    """
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # 1. Global summary / beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Global bar plot (mean absolute SHAP value)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                       plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_feature_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Local explanation (waterfall) for the first test sample
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0].values,
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_local_waterfall_sample0.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("SHAP plots saved to", save_dir)
    return shap_values


# ----------------------------- LIME -----------------------------------

def run_lime_analysis(model, X_train, X_test, feature_names, target_names,
                       sample_index=0, save_dir="../outputs/figures"):
    """
    Uses LIME to explain a single prediction by locally approximating the
    model with an interpretable linear model around that instance.
    """
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=target_names,
        mode="classification",
        discretize_continuous=True,
    )

    instance = X_test.iloc[sample_index].values
    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10,
    )

    # Save as a matplotlib figure
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 6)
    plt.title(f"LIME Explanation — Test Sample #{sample_index}")
    plt.tight_layout()
    fig.savefig(f"{save_dir}/lime_explanation_sample{sample_index}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also save the human-readable HTML explanation (nice for a portfolio/demo)
    exp.save_to_file(f"{save_dir}/lime_explanation_sample{sample_index}.html")

    print("LIME plot + HTML saved to", save_dir)
    return exp
