"""
main.py
Entry point: orchestrates data loading -> preprocessing -> training -> evaluation.
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_movies_data
from src.preprocess import preprocess
from src.train import train_and_evaluate


def main():
    print("=" * 60)
    print("  Movie Rating Prediction Pipeline")
    print("=" * 60)

    print("\n[Step 1] Loading data ...")
    df = load_movies_data(local_path="data/movies.csv")

    print("\n[Step 2] Preprocessing ...")
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df,
        test_size=0.2,
        random_state=42,
        preprocessor_path="models/preprocessor.pkl",
    )

    print("\n[Step 3] Training & evaluating models ...")
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        feature_names=feature_names,
        viz_dir="visualizations",
        model_dir="models",
    )

    print("\n" + "=" * 60)
    print("  Final Results Summary")
    print("=" * 60)
    header = f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<20} "
            f"{metrics['mae']:>8.4f} "
            f"{metrics['rmse']:>8.4f} "
            f"{metrics['r2']:>8.4f}"
        )

    best = max(results, key=lambda n: results[n]["r2"])
    print(f"\n  Best model by R2: {best}  ({results[best]['r2']:.4f})")
    print("\n  Artefacts saved:")
    print("    models/best_model.pkl")
    print("    models/preprocessor.pkl")
    print("    models/metrics.json")
    print("    visualizations/*.png")
    print("\n[Pipeline complete]")


if __name__ == "__main__":
    main()