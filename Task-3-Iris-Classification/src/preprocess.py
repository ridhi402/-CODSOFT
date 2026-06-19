"""
preprocess.py
Cleans the raw iris DataFrame, encodes the target label, scales features,
and splits into train/test sets.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, keyword: str) -> str:
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    raise KeyError(f"No column found containing '{keyword}'. Available: {list(df.columns)}")


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessor_path: str = "models/preprocessor.pkl",
):
    """
    Full preprocessing pipeline. Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    df = _clean_column_names(df)

    # Identify target column (could be "species", "class", etc.)
    target_col = None
    for kw in ["species", "class"]:
        try:
            target_col = _find_col(df, kw)
            break
        except KeyError:
            continue
    if target_col is None:
        raise KeyError("Could not find a target column (species/class).")

    # Identify the 4 numeric feature columns (everything except an id/target column,
    # and anything non-numeric)
    feature_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Drop an "Id" column if present (common in this dataset's csv)
    feature_cols = [c for c in feature_cols if "id" != c.lower()]

    print(f"  Target column: {target_col}")
    print(f"  Feature columns: {feature_cols}")

    # Drop any rows with missing values (Iris is usually clean, but just in case)
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"  Rows after dropping missing values: {len(df)}")

    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Encode target labels (e.g. "Iris-setosa" -> 0, "Iris-versicolor" -> 1, ...)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print(f"  Classes encoded: {dict(zip(class_names, range(len(class_names))))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = feature_cols

    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "class_names": class_names,
        },
        preprocessor_path,
    )
    print(f"  Preprocessor saved to '{preprocessor_path}'.")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, class_names