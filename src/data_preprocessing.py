"""
data_preprocessing.py
----------------------
Loads the Scikit-learn Breast Cancer Wisconsin (Diagnostic) dataset,
converts it to a pandas DataFrame, and performs the train/test split
and feature scaling needed before model training.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the breast cancer dataset as a pandas DataFrame.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (30 numeric features computed from digitized
        images of a fine needle aspirate (FNA) of a breast mass).
    y : pd.Series
        Target vector (0 = malignant, 1 = benign).
    feature_names : list[str]
    target_names : list[str]
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, list(data.feature_names), list(data.target_names)


def split_and_scale(X, y, test_size=0.2, random_state=11):
    """Train/test split followed by standardization (zero mean, unit variance).

    Logistic Regression is sensitive to feature scale, so we fit the
    scaler ONLY on the training data and use it to transform both
    train and test sets (to avoid data leakage).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Keep as DataFrames so column names are preserved (SHAP/LIME need them)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    print("Feature matrix shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
    print("Classes:", target_names)  # ['malignant' 'benign']
