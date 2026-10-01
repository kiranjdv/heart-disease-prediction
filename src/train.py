"""
train.py
Trains several classifiers on the heart disease dataset, evaluates them,
and saves the best-performing model + scaler to the models/ folder.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from preprocess import load_data, clean_data, split_data, scale_features

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC_AUC": roc_auc_score(y_test, probs),
        "CV_Mean": cv_scores.mean(),
        "CV_Std": cv_scores.std(),
    }
    return model, metrics, confusion_matrix(y_test, preds)


def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    results = []
    trained_models = {}
    confusions = {}

    for name, model in MODELS.items():
        trained_model, metrics, cm = evaluate_model(
            name, model, X_train_s, X_test_s, y_train, y_test
        )
        results.append(metrics)
        trained_models[name] = trained_model
        confusions[name] = cm

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    print(f"\nBest model: {best_name}")
    print("Confusion matrix:\n", confusions[best_name])

    # Save best model, scaler, and feature order
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")
    results_df.to_csv("models/model_comparison.csv", index=False)

    print("\nSaved best_model.pkl, scaler.pkl, feature_columns.pkl to models/")


if __name__ == "__main__":
    main()
