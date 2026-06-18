"""
train.py
Trains multiple regression models, tunes hyperparameters, evaluates on a
held-out test set, saves the best model + metrics + visualizations.
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    feature_names,
    viz_dir: str = "visualizations",
    model_dir: str = "models",
):
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    trained = {}
    results = {}

    for name, model in models.items():
        if name in PARAM_GRIDS:
            print(f"  GridSearchCV -> {name} ...")
            gs = GridSearchCV(
                model, PARAM_GRIDS[name], cv=5,
                scoring="neg_mean_squared_error", n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            print(f"    Best params : {gs.best_params_}")
            print(f"    Best CV RMSE: {np.sqrt(-gs.best_score_):.4f}")
        else:
            print(f"  Training (no grid) -> {name} ...")
            best = model
            best.fit(X_train, y_train)
            cv_scores = cross_val_score(
                best, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
            )
            print(f"    CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} ± {cv_scores.std():.4f}")

        trained[name] = best

        # --- Test set evaluation ---
        y_pred = best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"    Test  -> MAE: {mae:.4f}  RMSE: {rmse:.4f}  R2: {r2:.4f}")

    # --- Pick best model by R2 ---
    best_name = max(results, key=lambda n: results[n]["r2"])
    best_model = trained[best_name]
    print(f"\n  Best model: {best_name} (R2 = {results[best_name]['r2']:.4f})")

    # --- Save model, metrics ---
    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Visualizations for the best model ---
    y_pred_best = best_model.predict(X_test)

    # Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.5, edgecolor="k")
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Actual vs Predicted Rating ({best_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "actual_vs_predicted.png"))
    plt.close()

    # Residual plot
    residuals = y_test - y_pred_best
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred_best, residuals, alpha=0.5, edgecolor="k")
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted Rating")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residual Plot ({best_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "residual_plot.png"))
    plt.close()

    # Feature importances (only for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        order = np.argsort(importances)[::-1][:15]  # top 15
        plt.figure(figsize=(8, 6))
        plt.barh(
            [feature_names[i] for i in order][::-1],
            importances[order][::-1],
        )
        plt.xlabel("Importance")
        plt.title(f"Top Feature Importances ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "feature_importances.png"))
        plt.close()

    print(f"  Visualizations saved to '{viz_dir}/'.")

    return results