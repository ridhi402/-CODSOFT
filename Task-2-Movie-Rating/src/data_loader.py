"""
data_loader.py
Downloads the IMDb India Movies dataset via kagglehub and saves a local copy.
"""

import os
import shutil
import pandas as pd


def load_movies_data(local_path: str = "data/movies.csv") -> pd.DataFrame:
    """
    Downloads the dataset (if not already cached locally) and returns it as a
    pandas DataFrame. Also saves a copy to `local_path` for easy access.
    """
  
    if os.path.exists(local_path):
        print(f"  Found existing local copy at '{local_path}', loading from disk.")
        df = pd.read_csv(local_path, encoding="latin1")
        _validate(df)
        return df

    print("  Downloading dataset via kagglehub ...")
    import kagglehub

    path = kagglehub.dataset_download("adrianmcmahon/imdb-india-movies")
    print(f"  Dataset downloaded to: {path}")

  
    csv_file = None
    for f in os.listdir(path):
        if f.lower().endswith(".csv"):
            csv_file = os.path.join(path, f)
            break

    if csv_file is None:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset path: {path}")

   
    df = pd.read_csv(csv_file, encoding="latin1")

    # Save a local copy for next time
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

    rating_cols = [c for c in df.columns if "rating" in c.lower()]
    if not rating_cols:
        print("  WARNING: Could not find a 'Rating' column. Check column names above.")
    else:
        print(f"  Target column detected: {rating_cols[0]}")