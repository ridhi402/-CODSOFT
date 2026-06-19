"""
data_loader.py
Downloads the Iris Flower dataset via kagglehub and saves a local copy.
"""

import os
import pandas as pd


def load_iris_data(local_path: str = "data/iris.csv") -> pd.DataFrame:
    """
    Downloads the dataset (if not already cached locally) and returns it as a
    pandas DataFrame. Also saves a copy to `local_path` for easy access.
    """
    if os.path.exists(local_path):
        print(f"  Found existing local copy at '{local_path}', loading from disk.")
        df = pd.read_csv(local_path)
        _validate(df)
        return df

    print("  Downloading dataset via kagglehub ...")
    import kagglehub

    path = kagglehub.dataset_download("arshid/iris-flower-dataset")
    print(f"  Dataset downloaded to: {path}")

    csv_file = None
    for f in os.listdir(path):
        if f.lower().endswith(".csv"):
            csv_file = os.path.join(path, f)
            break

    if csv_file is None:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset path: {path}")

    df = pd.read_csv(csv_file)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"  Local copy saved to '{local_path}'.")

    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Basic sanity checks on the loaded dataset."""
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    if df.empty:
        raise ValueError("Loaded dataset is empty!")

    species_cols = [c for c in df.columns if "species" in c.lower() or "class" in c.lower()]
    if not species_cols:
        print("  WARNING: Could not find a 'species'/'class' column. Check column names above.")
    else:
        print(f"  Target column detected: {species_cols[0]}")
        print(f"  Classes: {df[species_cols[0]].unique().tolist()}")