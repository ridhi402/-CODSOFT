"""
main.py
Entry point: orchestrates data loading → preprocessing → training → evaluation.
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_titanic_data
from src.preprocess import preprocess
from src.train import train_and_evaluate


def main():
    print("=" * 60)
    print("  Titanic Survival Prediction Pipeline")
    print("=" * 60)

    print("\n[Step 1] Loading data …")
    df = load_titanic_data(local_path="data/titanic.csv")

    print("\n[Step 2] Preprocessing …")
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df,
        test_size=0.2,
        random_state=42,
        preprocessor_path="models/preprocessor.pkl",
    )

    print("\n[Step 3] Training & evaluating models …")
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        feature_names=feature_names,
        viz_dir="visualizations",
        model_dir="models",
    )


    print("\n" + "=" * 60)
    print("  Final Results Summary")
    print("=" * 60)
    header = f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['accuracy']:>6.4f} "
            f"{metrics['precision']:>6.4f} "
            f"{metrics['recall']:>6.4f} "
            f"{metrics['f1']:>6.4f} "
            f"{metrics['roc_auc']:>6.4f}"
        )

    best = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\n  Best model by ROC-AUC: {best}  ({results[best]['roc_auc']:.4f})")
    print("\n  Artefacts saved:")
    print("    models/best_model.pkl")
    print("    models/preprocessor.pkl")
    print("    models/metrics.json")
    print("    visualizations/*.png")
    print("\n[Pipeline complete] ✓")


if __name__ == "__main__":
    main()