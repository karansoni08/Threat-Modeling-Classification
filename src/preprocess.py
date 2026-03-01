from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess(df: pd.DataFrame, target_col: str = "Label"):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Columns (sample): {list(df.columns)[:30]}"
        )

    # Replace common bad values
    df.replace(["Infinity", "inf", "INF", "NaN", "nan"], np.nan, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows without label
    df = df.dropna(subset=[target_col]).copy()

    y_raw = df[target_col].astype(str)
    X = df.drop(columns=[target_col])

    # Convert everything to numeric where possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, le, list(X.columns)
