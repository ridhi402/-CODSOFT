"""
train.py
Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
Performs GridSearchCV tuning, evaluates on held-out test set, and saves artefacts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report,
)
import joblib



PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, None],
        "min_samples_split": [2, 5],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
    },
}



def _get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }



def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    feature_names: list,
    viz_dir: str = "visualizations",
    model_dir: str = "models",
) -> dict:
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    models = _get_models()
    results = {}
    trained = {}

    print("\n[train] ── Cross-validation & tuning ──")
    for name, model in models.items():
        if name in PARAM_GRIDS:
            print(f"  GridSearchCV → {name} …")
            gs = GridSearchCV(
                model, PARAM_GRIDS[name], cv=5, scoring="accuracy", n_jobs=-1
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            print(f"    Best params : {gs.best_params_}")
            print(f"    Best CV acc : {gs.best_score_:.4f}")
        else:
            print(f"  Training (no grid) → {name} …")
            best = model
            best.fit(X_train, y_train)
            cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="accuracy")
            print(f"    CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        trained[name] = best

    # ── Test-set evaluation ──────────────────────────────────────────────────
    print("\n[train] ── Test-set evaluation ──")
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        }
        results[name] = metrics

        print(f"\n  {name}")
        for k, v in metrics.items():
            print(f"    {k:<12}: {v}")
        print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

        # Confusion matrix
        _plot_confusion_matrix(y_test, y_pred, name, viz_dir)

        # ROC curve
        _plot_roc_curve(y_test, y_prob, name, metrics["roc_auc"], viz_dir)

    # Feature importances (tree-based models) 
    for name in ("RandomForest", "GradientBoosting"):
        _plot_feature_importances(trained[name], feature_names, name, viz_dir)

    # Pick best model by ROC-AUC 
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = trained[best_name]
    best_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, best_path)
    print(f"\n[train] Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")
    print(f"[train] Saved to '{best_path}'.")

    # Save metrics JSON
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"[train] Metrics saved to '{metrics_path}'.")

    return results


def _plot_confusion_matrix(y_test, y_pred, model_name, viz_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Died", "Survived"],
        yticklabels=["Died", "Survived"], ax=ax,
    )
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(viz_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Saved confusion matrix → {path}")


def _plot_roc_curve(y_test, y_prob, model_name, auc, viz_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="steelblue", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(viz_dir, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Saved ROC curve → {path}")


def _plot_feature_importances(model, feature_names, model_name, viz_dir, top_n=15):
    importances = model.feature_importances_
    # Pad or trim in case of length mismatch
    n = min(len(importances), len(feature_names))
    imp_series = pd.Series(importances[:n], index=feature_names[:n])
    imp_series = imp_series.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=imp_series.values, y=imp_series.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importances – {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(viz_dir, f"feature_importances_{model_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Saved feature importances → {path}")