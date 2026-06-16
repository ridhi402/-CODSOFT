"""
data_loader.py
Downloads the Titanic dataset via kagglehub and saves a local copy
"""

import os
import shutil
import pandas as pd


def load_titanic_data(local_path: str = "data/titanic.csv") -> pd.DataFrame:
    """
    Attempt to load the Titanic dataset.
    1. If a local CSV already exists, load it directly.
    2. Otherwise try kagglehub; if credentials are missing fall back to
       the well-known seaborn / GitHub mirror so the pipeline always works.
    """
    if os.path.exists(local_path):
        print(f"[data_loader] Loading existing local dataset from '{local_path}'.")
        df = pd.read_csv(local_path)
        _validate(df)
        return df

   
    try:
        import kagglehub
        print("[data_loader] Downloading dataset via kagglehub …")
        dataset_path = kagglehub.dataset_download("yasserh/titanic-dataset")
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in kagglehub download.")
        src = os.path.join(dataset_path, csv_files[0])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy(src, local_path)
        print(f"[data_loader] Dataset saved to '{local_path}'.")
    except Exception as e:
        print(f"[data_loader] kagglehub unavailable ({e}). Falling back to public mirror …")
        url = (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/"
            "master/titanic.csv"
        )
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)
        print(f"[data_loader] Fallback dataset saved to '{local_path}'.")
        _validate(df)
        return df

    df = pd.read_csv(local_path)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    required = {"Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[data_loader] Dataset is missing expected columns: {missing}")
    print(f"[data_loader] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns.")
    print(f"[data_loader] Survival rate: {df['Survived'].mean():.2%}")