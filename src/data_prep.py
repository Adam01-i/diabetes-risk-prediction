"""
Chargement et préparation du jeu de données Pima Indians Diabetes.

Problème de qualité de données traité ici (le principal piège de ce
dataset, connu mais absent de tout code fourni avec le fichier) : les
colonnes `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` et `BMI`
utilisent la valeur **0 comme encodage de valeur manquante**, et non comme
une vraie mesure (une glycémie, une tension artérielle ou un IMC de 0 chez
une personne vivante n'est pas physiologiquement possible). Sans ce
traitement, un modèle entraîné directement sur les données brutes
apprendrait des zéros artificiels comme s'il s'agissait de vraies mesures
basses — un biais silencieux qui n'apparaît dans aucune erreur ni warning.

Répartition des zéros suspects dans le dataset (768 lignes) :
- Insulin : 48.7% de zéros
- SkinThickness : 29.6% de zéros
- BloodPressure : 4.6% de zéros
- BMI : 1.4% de zéros
- Glucose : 0.7% de zéros

La colonne `Age` contient par ailleurs quelques vraies valeurs manquantes
(NaN, 3 lignes) à traiter séparément.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Outcome"
RANDOM_STATE = 42

# Colonnes où 0 est physiologiquement impossible -> encodage de valeur manquante
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes attendues manquantes dans {path}: {missing}")
    return df


def replace_zero_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les 0 non plausibles par NaN, pour qu'ils soient ensuite
    traités comme de vraies valeurs manquantes (imputation) plutôt que
    comme des mesures réelles."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        n_zeros = (df[col] == 0).sum()
        if n_zeros:
            print(f"{col} : {n_zeros} zéros ({n_zeros/len(df)*100:.1f}%) traités comme valeurs manquantes.")
        df[col] = df[col].replace(0, np.nan).astype(float)
    return df


def impute_with_train_stats(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Impute par la médiane calculée UNIQUEMENT sur le train, puis
    appliquée au train et au test -> pas de fuite de données (la médiane du
    test ne doit jamais influencer l'imputation)."""
    medians = X_train.median()
    X_train_imputed = X_train.fillna(medians)
    X_test_imputed = X_test.fillna(medians)
    return X_train_imputed, X_test_imputed, medians


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Pipeline complet : 0->NaN, split stratifié, imputation (train only),
    standardisation (train only). Retourne les jeux prêts pour l'entraînement."""
    df = replace_zero_with_nan(df)
    df["Age"] = df["Age"].fillna(df["Age"].median())  # 3 NaN réels, peu impactant

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test, medians = impute_with_train_stats(X_train, X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, medians, X.columns.tolist()
