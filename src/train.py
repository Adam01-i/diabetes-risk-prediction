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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from data_prep import RANDOM_STATE, load_dataset, prepare_data

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


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

    print(f"\nModèle sauvegardé dans {MODELS_DIR / 'model.pkl'}")
    print(f"Métriques sauvegardées dans {OUTPUTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
