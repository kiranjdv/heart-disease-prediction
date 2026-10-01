"""
evaluate.py
Generates EDA plots and feature-importance charts for the report/presentation.
Run this after train.py. Saves all figures into models/figures/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, clean_data

FIG_DIR = "models/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=150)
    plt.close()


def eda(df):
    # Target class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="target", data=df, palette="Set2")
    plt.title("Heart Disease Class Balance (0 = No Disease, 1 = Disease)")
    save("class_balance.png")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    save("correlation_heatmap.png")

    # Age distribution by target
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="age", hue="target", kde=True, palette="Set1", multiple="stack")
    plt.title("Age Distribution by Heart Disease Status")
    save("age_distribution.png")

    # Cholesterol vs target
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="target", y="chol", data=df, palette="Set2")
    plt.title("Cholesterol Levels by Heart Disease Status")
    save("cholesterol_boxplot.png")

    # Max heart rate vs target
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="target", y="thalach", data=df, palette="Set3")
    plt.title("Max Heart Rate Achieved by Heart Disease Status")
    save("thalach_boxplot.png")


def feature_importance(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    save("feature_importance.png")

    print("\nFeature importance ranking:")
    print(importances)


def main():
    df = load_data()
    df = clean_data(df)
    eda(df)
    feature_importance(df)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
