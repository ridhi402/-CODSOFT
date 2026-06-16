"""
preprocess.py
Cleans, engineers features, encodes, and scales the Titanic dataset.
Returns train/test splits ready for modelling.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os


# ── Title extraction ─────────────────────────────────────────────────────────

_TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Capt": "Rare",
    "Sir": "Rare", "Mme": "Mrs",
}


def _extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    if match:
        raw = match.group(1).strip()
        return _TITLE_MAP.get(raw, "Rare")
    return "Rare"


# ── Core engineering ─────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Title
    df["Title"] = df["Name"].apply(_extract_title)

    # Age imputation: median by Pclass + Sex
    age_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_medians)
    # Final fallback for any remaining NaN
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked imputation
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fare imputation (rare but possible in test sets)
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Cabin → binary
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    return df


# ── Preprocessing pipeline ───────────────────────────────────────────────────

NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
PASSTHROUGH_FEATURES = ["Pclass", "HasCabin", "IsAlone"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("pass", "passthrough", PASSTHROUGH_FEATURES),
    ])


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessor_path: str = "models/preprocessor.pkl",
) -> tuple:
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    df = engineer_features(df)

    X = df[ALL_FEATURES].copy()
    y = df["Survived"].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Derive feature names for interpretability
    ohe = preprocessor.named_transformers_["cat"]["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names + PASSTHROUGH_FEATURES

    # Save preprocessor
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[preprocess] Preprocessor saved to '{preprocessor_path}'.")

    print(f"[preprocess] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    assert not np.isnan(X_train).any(), "NaNs remain in X_train!"
    assert not np.isnan(X_test).any(), "NaNs remain in X_test!"
    print("[preprocess] No NaNs in processed arrays. ✓")

    return X_train, X_test, y_train, y_test, feature_names