"""
train_model.py
---------------
Trains a Logistic Regression classifier on the breast cancer dataset.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_data, split_and_scale


def train_logistic_regression(X_train, y_train, random_state=11):
    model = LogisticRegression(max_iter=5000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    model = train_logistic_regression(X_train, y_train)

    joblib.dump(model, "../models/logistic_regression_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    print("Model and scaler saved to ../models/")
