"""
preprocess.py
Cleans the raw movies DataFrame, engineers features, encodes categoricals,
scales numerics, and splits into train/test sets.
"""

import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip whitespace, consistent casing."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, keyword: str) -> str:
    """Find a column whose name contains `keyword` (case-insensitive)."""
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    raise KeyError(f"No column found containing '{keyword}'. Available: {list(df.columns)}")


def _parse_year(series: pd.Series) -> pd.Series:
    """Extract a 4-digit year from strings like '(2019)' -> 2019."""
    return series.astype(str).str.extract(r"(\d{4})").astype(float)


def _parse_duration(series: pd.Series) -> pd.Series:
    """Extract numeric minutes from strings like '109 min' -> 109."""
    return series.astype(str).str.extract(r"(\d+)").astype(float)


def _parse_votes(series: pd.Series) -> pd.Series:
    """Strip commas from vote counts like '1,234' -> 1234."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")
        .astype(float)
    )


def _frequency_encode(train_series: pd.Series, test_series: pd.Series):
    """
    Encode high-cardinality categorical columns (Director, Actors) by how
    often each value appears in the TRAINING set. Unseen categories in test
    get a frequency of 0.
    """
    freq_map = train_series.value_counts(normalize=True)
    train_encoded = train_series.map(freq_map).fillna(0)
    test_encoded = test_series.map(freq_map).fillna(0)
    return train_encoded, test_encoded, freq_map


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessor_path: str = "models/preprocessor.pkl",
):
    """
    Full preprocessing pipeline. Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = _clean_column_names(df)

    name_col = _find_col(df, "name")
    year_col = _find_col(df, "year")
    duration_col = _find_col(df, "duration")
    genre_col = _find_col(df, "genre")
    rating_col = _find_col(df, "rating")
    votes_col = _find_col(df, "votes")
    director_col = _find_col(df, "director")

    actor_cols = [c for c in df.columns if "actor" in c.lower()]

    print(f"  Using columns -> Year: {year_col}, Duration: {duration_col}, "
          f"Genre: {genre_col}, Rating: {rating_col}, Votes: {votes_col}, "
          f"Director: {director_col}, Actors: {actor_cols}")

    df = df.dropna(subset=[rating_col]).copy()
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])
    print(f"  Rows remaining after dropping missing ratings: {len(df)}")

    df["Year_clean"] = _parse_year(df[year_col])
    df["Duration_clean"] = _parse_duration(df[duration_col])
    df["Votes_clean"] = _parse_votes(df[votes_col])

    df["Year_clean"] = df["Year_clean"].fillna(df["Year_clean"].median())
    df["Duration_clean"] = df["Duration_clean"].fillna(df["Duration_clean"].median())
    df["Votes_clean"] = df["Votes_clean"].fillna(df["Votes_clean"].median())

    df["Votes_log"] = np.log1p(df["Votes_clean"])

    df["Primary_Genre"] = (
        df[genre_col].astype(str).str.split(",").str[0].str.strip().replace("nan", "Unknown")
    )

    if actor_cols:
        actor_df = df[actor_cols].fillna("Unknown").astype(str)
        df["Cast_combined"] = actor_df.agg(", ".join, axis=1)
    else:
        df["Cast_combined"] = "Unknown"

    df[director_col] = df[director_col].fillna("Unknown")

    feature_cols_raw = ["Year_clean", "Duration_clean", "Votes_log",
                         "Primary_Genre", director_col, "Cast_combined"]
    X = df[feature_cols_raw].copy()
    y = df[rating_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train["Director_freq"], X_test["Director_freq"], director_freq_map = _frequency_encode(
        X_train[director_col], X_test[director_col]
    )
    X_train["Cast_freq"], X_test["Cast_freq"], cast_freq_map = _frequency_encode(
        X_train["Cast_combined"], X_test["Cast_combined"]
    )

    X_train_genre = pd.get_dummies(X_train["Primary_Genre"], prefix="Genre")
    X_test_genre = pd.get_dummies(X_test["Primary_Genre"], prefix="Genre")
    X_test_genre = X_test_genre.reindex(columns=X_train_genre.columns, fill_value=0)

    numeric_cols = ["Year_clean", "Duration_clean", "Votes_log", "Director_freq", "Cast_freq"]

    X_train_final = pd.concat(
        [X_train[numeric_cols].reset_index(drop=True),
         X_train_genre.reset_index(drop=True)],
        axis=1,
    )
    X_test_final = pd.concat(
        [X_test[numeric_cols].reset_index(drop=True),
         X_test_genre.reset_index(drop=True)],
        axis=1,
    )

    scaler = StandardScaler()
    X_train_final[numeric_cols] = scaler.fit_transform(X_train_final[numeric_cols])
    X_test_final[numeric_cols] = scaler.transform(X_test_final[numeric_cols])

    feature_names = list(X_train_final.columns)

    import os
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "director_freq_map": director_freq_map,
            "cast_freq_map": cast_freq_map,
            "genre_columns": list(X_train_genre.columns),
            "numeric_cols": numeric_cols,
            "feature_names": feature_names,
        },
        preprocessor_path,
    )
    print(f"  Preprocessor saved to '{preprocessor_path}'.")
    print(f"  Final feature count: {len(feature_names)}")

    return (
        X_train_final.values,
        X_test_final.values,
        y_train.values,
        y_test.values,
        feature_names,
    )