"""
preprocess.py
Loads the raw heart disease dataset, cleans it, and splits it into
train/test sets ready for modeling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path="data/heart.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean_data(df):
    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Drop rows with any missing values (dataset is small & mostly clean,
    # so imputation isn't necessary here)
    df = df.dropna()

    return df.reset_index(drop=True)


def split_data(df, target_col="target", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    print("Shape after cleaning:", df.shape)
    print(df["target"].value_counts())
