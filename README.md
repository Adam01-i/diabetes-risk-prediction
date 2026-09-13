# Diabetes Risk Prediction

Pipeline de machine learning pour estimer le risque de diabète à partir de
mesures cliniques. Le projet compare une régression logistique interprétable
à un modèle `RandomForestClassifier`, tout en traitant explicitement les
valeurs manquantes cachées du jeu de données Pima Indians Diabetes.

> **Avertissement** : ce projet est éducatif. Les prédictions ne constituent
> ni un diagnostic, ni un avis médical, ni une recommandation de prise en
> charge. Le modèle n'est pas validé pour un usage clinique.

## Résultats clés

Évaluation sur un jeu de test stratifié représentant 20 % des 768 lignes,
avec `random_state=42`. Les métriques ci-dessous sont celles générées par
la dernière exécution de `src/train.py` et enregistrées dans
`outputs/metrics.json`.

| Modèle | Accuracy | Précision classe 1 | Rappel classe 1 | F1 classe 1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Régression logistique | 0,734 | 0,603 | 0,704 | 0,650 | 0,812 |
| **Random forest** | **0,760** | **0,644** | **0,704** | **0,673** | **0,831** |

Le random forest est sélectionné automatiquement selon le ROC AUC, puis
sauvegardé dans `models/model.pkl`. Les deux modèles utilisent
`class_weight="balanced"` pour tenir compte des 34,9 % de cas positifs.

## Démarrage rapide

### 1. Installer l'environnement

Python 3.12 est recommandé.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Entraîner les modèles

```bash
python src/train.py --data data/diabetes.csv
```

Cette commande compare les deux modèles, affiche leurs métriques et produit :

- `models/model.pkl` : modèle retenu et éléments du prétraitement ;
- `outputs/metrics.json` : rapports de classification, matrices de confusion
	et ROC AUC.

### 3. Prédire sur de nouveaux patients

```bash
python src/predict.py --patients data/new_patients_example.csv
```

Le fichier de modèle doit exister avant cette étape. Un chemin différent
peut être fourni avec `--model` :

```bash
python src/predict.py \
	--patients data/new_patients_example.csv \
	--model models/model.pkl
```

La sortie indique la classe prédite (`1` ou `0`) et la probabilité estimée
de la classe positive pour chaque ligne.

## Format des données

### Entraînement

`data/diabetes.csv` contient 768 patientes et les colonnes suivantes :

| Colonne | Description |
| --- | --- |
| `Pregnancies` | Nombre de grossesses |
| `Glucose` | Concentration de glucose |
| `BloodPressure` | Pression artérielle diastolique |
| `SkinThickness` | Épaisseur du pli cutané |
| `Insulin` | Insulinémie |
| `BMI` | Indice de masse corporelle |
| `DiabetesPedigreeFunction` | Fonction de pedigree du diabète |
| `Age` | Âge |
| `Outcome` | Cible : `0` ou `1` |

### Prédiction

Un fichier fourni à `--patients` doit contenir les huit variables
explicatives, dans n'importe quel ordre. Il ne doit pas contenir la colonne
`Outcome`.

Exemple minimal de structure :

```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
6,148,72,35,0,33.6,0.627,50
1,85,66,29,0,26.6,0.351,31
```

## Prétraitement et prévention de la fuite de données

Le pipeline de `src/data_prep.py` applique les mêmes règles à
l'entraînement et à la prédiction :

1. Les zéros de `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` et
	 `BMI` sont considérés comme des valeurs manquantes. Dans ce dataset, ils
	 représentent notamment 48,7 % des valeurs de `Insulin` et 29,6 % de
	 celles de `SkinThickness`.
2. Les valeurs manquantes sont imputées par la médiane.
3. Le jeu est séparé avec stratification avant l'imputation destinée au
	 modèle.
4. Les médianes et le `StandardScaler` sont calculés à partir du train
	 uniquement.
5. Ces statistiques sont stockées dans `model.pkl` et réutilisées pour les
	 nouveaux patients.

Cette organisation évite que les statistiques du jeu de test ou des
patients à prédire influencent l'entraînement.

## Structure du dépôt

```text
diabetes-risk-prediction/
├── data/
│   ├── diabetes.csv                  # jeu d'entraînement
│   └── new_patients_example.csv      # exemples de prédiction
├── models/                           # artefacts générés
├── outputs/
│   └── metrics.json                  # métriques générées
├── src/
│   ├── data_prep.py                  # nettoyage et préparation
│   ├── train.py                      # entraînement et évaluation
│   └── predict.py                    # prédiction sur un CSV
├── requirements.txt
├── LICENSE
└── README.md
```

## Interprétation

Les coefficients de la régression logistique, calculés sur les variables
standardisées, donnent un repère d'interprétation du modèle linéaire. Dans
l'exécution de référence, les contributions positives les plus fortes sont
associées à `Glucose` (1,184), `BMI` (0,710) et `Pregnancies` (0,373).
Ces coefficients ne sont pas des causalités médicales et ne doivent pas
être interprétés séparément du contexte clinique.

## Limites

- Le jeu de données est petit et provient d'une population spécifique ; sa
	représentativité pour d'autres populations n'est pas établie.
- L'évaluation repose sur un seul découpage train/test et ne remplace pas
	une validation externe ou une validation croisée complète.
- Les probabilités affichées n'ont pas fait l'objet d'une étude de
	calibration.
- Le seuil de décision par défaut du classifieur n'a pas été optimisé pour
	un cas d'usage clinique particulier.
- Aucune décision médicale ne doit être automatisée à partir de ce dépôt.

## Dépendances

- Python 3
- pandas
- NumPy
- scikit-learn

Les versions minimales sont précisées dans `requirements.txt`.

## Licence

Projet distribué sous licence MIT. Voir [LICENSE](LICENSE).
