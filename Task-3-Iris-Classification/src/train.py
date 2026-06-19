"""
train.py
Trains multiple classifiers, tunes hyperparameters, evaluates on a held-out
test set, saves the best model + metrics + visualizations.
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)


PARAM_GRIDS = {
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    },
}


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    feature_names, class_names,
    viz_dir: str = "visualizations",
    model_dir: str = "models",
):
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "RandomForest": RandomForestClassifier(random_state=42),
    }

    trained = {}
    results = {}

    for name, model in models.items():
        if name in PARAM_GRIDS:
            print(f"  GridSearchCV -> {name} ...")
            gs = GridSearchCV(
                model, PARAM_GRIDS[name], cv=5, scoring="accuracy", n_jobs=-1
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            print(f"    Best params : {gs.best_params_}")
            print(f"    Best CV acc : {gs.best_score_:.4f}")
        else:
            print(f"  Training (no grid) -> {name} ...")
            best = model
            best.fit(X_train, y_train)
            cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="accuracy")
            print(f"    CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        trained[name] = best

        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"    Test -> Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = trained[best_name]
    print(f"\n  Best model: {best_name} (Accuracy = {results[best_name]['accuracy']:.4f})")

    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Confusion Matrix ---
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confusion_matrix.png"))
    plt.close()

    # --- Feature Importances (tree-based only) ---
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(7, 5))
        plt.barh(
            [feature_names[i] for i in order][::-1],
            importances[order][::-1],
        )
        plt.xlabel("Importance")
        plt.title(f"Feature Importances ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "feature_importances.png"))
        plt.close()

    # --- Pairplot of the original features colored by species (great for Iris) ---
    try:
        import pandas as pd
        df_viz = pd.DataFrame(X_test, columns=feature_names)
        df_viz["species"] = [class_names[i] for i in y_test]
        sns.pairplot(df_viz, hue="species", diag_kind="hist")
        plt.savefig(os.path.join(viz_dir, "feature_pairplot.png"))
        plt.close()
    except Exception as e:
        print(f"  (Skipped pairplot: {e})")

    print(f"  Visualizations saved to '{viz_dir}/'.")

    return results