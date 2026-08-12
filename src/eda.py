"""
eda.py
------
Basic exploratory data analysis: class balance and feature correlation,
saved as figures for the README / notebook.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data


def plot_class_distribution(y, target_names, save_path):
    plt.figure(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    labels = [target_names[i] for i in counts.index]
    colors = ["#e74c3c", "#2ecc71"]
    plt.bar(labels, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        plt.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.title("Class Distribution: Benign vs Malignant")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(X, save_path, top_n=15):
    # Show correlation among the most variable features for readability
    corr = X.corr()
    top_features = X.var().sort_values(ascending=False).head(top_n).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr.loc[top_features, top_features], annot=True, fmt=".2f",
                cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Feature Correlation Heatmap (Top {top_n} Highest-Variance Features)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    plot_class_distribution(y, target_names, "../outputs/figures/class_distribution.png")
    plot_correlation_heatmap(X, "../outputs/figures/correlation_heatmap.png")
    print("EDA figures saved.")
