"""
Charge le modèle entraîné (models/model.pkl) et prédit le risque de
diabète pour de nouveaux patients fournis dans un CSV.

Le CSV d'entrée doit contenir les mêmes colonnes explicatives que les
données d'entraînement (toutes sauf Outcome). Les zéros non plausibles
(Glucose, BloodPressure, SkinThickness, Insulin, BMI) sont automatiquement
traités comme des valeurs manquantes et imputés avec les médianes calculées
sur le jeu d'entraînement (sauvegardées dans le modèle), exactement comme
à l'entraînement — pas de nouvelle fuite de données.

Usage:
    python src/predict.py --patients data/new_patients_example.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from data_prep import ZERO_AS_MISSING_COLS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", required=True)
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    medians = bundle["medians"]
    feature_names = bundle["feature_names"]

    patients = pd.read_csv(args.patients)
    missing = set(feature_names) - set(patients.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {args.patients} : {missing}")

    X_new = patients[feature_names].copy()
    for col in ZERO_AS_MISSING_COLS:
        X_new[col] = X_new[col].replace(0, pd.NA)
    X_new = X_new.fillna(pd.Series(medians))

    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        label = "RISQUE ÉLEVÉ" if pred == 1 else "risque faible"
        print(f"Patient {i} -> {pred} ({label}, probabilité = {proba:.2f})")


if __name__ == "__main__":
    main()
