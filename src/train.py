"""
Entraînement de modèles de prédiction du diabète (Pima Indians Diabetes)
et comparaison Régression Logistique vs Random Forest.

Usage:
    python src/train.py --data data/diabetes.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import RANDOM_STATE, load_dataset, prepare_data

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def generate_figures(results, models, feature_names) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    labels = ["Régression logistique", "Random forest"]
    model_names = ["logistic_regression", "random_forest"]
    metric_names = ["accuracy", "precision", "recall", "f1-score", "roc_auc"]
    metric_labels = ["Accuracy", "Précision", "Rappel", "F1-score", "ROC AUC"]
    metric_values = [
        [results[name]["classification_report"]["accuracy"] for name in model_names],
        [results[name]["classification_report"]["1"]["precision"] for name in model_names],
        [results[name]["classification_report"]["1"]["recall"] for name in model_names],
        [results[name]["classification_report"]["1"]["f1-score"] for name in model_names],
        [results[name]["roc_auc"] for name in model_names],
    ]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metric_names))
    width = 0.36
    axis.bar(x - width / 2, [values[0] for values in metric_values], width, label=labels[0], color="#176b87")
    axis.bar(x + width / 2, [values[1] for values in metric_values], width, label=labels[1], color="#e07a5f")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Performance sur le jeu de test")
    axis.set_xticks(x, metric_labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, name, label in zip(axes, model_names, labels):
        matrix = np.array(results[name]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_title(label)
        axis.set_xlabel("Prédit")
        axis.set_ylabel("Réel")
        axis.set_xticks([0, 1], ["0", "1"])
        axis.set_yticks([0, 1], ["0", "1"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.colorbar(image, ax=axes, shrink=0.8, label="Nombre de patients")
    figure.suptitle("Matrices de confusion")
    figure.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=180)
    plt.close(figure)

    logistic_coefficients = models["logistic_regression"].coef_[0]
    order = np.argsort(np.abs(logistic_coefficients))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = ["#e07a5f" if logistic_coefficients[index] > 0 else "#176b87" for index in order]
    axis.barh(np.array(feature_names)[order], logistic_coefficients[order], color=colors)
    axis.axvline(0, color="#24323d", linewidth=0.8)
    axis.set_xlabel("Coefficient sur variables standardisées")
    axis.set_title("Facteurs les plus associés à la prédiction logistique")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "logistic_coefficients.png", dpi=180)
    plt.close(figure)


def evaluate(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :", cm)
    print(f"ROC AUC : {auc:.3f}")

    return {"classification_report": report, "confusion_matrix": cm, "roc_auc": auc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes.csv")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print(f"Chargement des données depuis {args.data} ...")
    df = load_dataset(args.data)
    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")
    print(f"Taux de diabète (Outcome=1) : {df['Outcome'].mean():.3f}")

    X_train, X_test, y_train, y_test, scaler, medians, feature_names = prepare_data(df)

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    results = {}
    best_name, best_model, best_auc = None, None, -1
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(name, model, X_test, y_test)
        if results[name]["roc_auc"] > best_auc:
            best_name, best_model, best_auc = name, model, results[name]["roc_auc"]

    print(f"\nMeilleur modèle : {best_name} (ROC AUC = {best_auc:.3f})")

    # Régression logistique : coefficients interprétables (importance des facteurs de risque)
    if hasattr(models["logistic_regression"], "coef_"):
        coefs = dict(zip(feature_names, models["logistic_regression"].coef_[0].round(3)))
        print("\nCoefficients de la régression logistique (facteurs de risque) :")
        for feat, coef in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            print(f"  {feat}: {coef}")

    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "model_name": best_name,
                "scaler": scaler,
                "medians": medians.to_dict(),
                "feature_names": feature_names,
            },
            f,
        )

    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    generate_figures(results, models, feature_names)

    print(f"\nModèle sauvegardé dans {MODELS_DIR / 'model.pkl'}")
    print(f"Métriques sauvegardées dans {OUTPUTS_DIR / 'metrics.json'}")
    print(f"Figures sauvegardées dans {FIGURES_DIR}")


if __name__ == "__main__":
    main()
