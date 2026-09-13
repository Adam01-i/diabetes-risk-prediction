# Prédiction du risque de diabète (Pima Indians Diabetes)

Pipeline de classification binaire pour prédire le risque de diabète à
partir de mesures cliniques (glycémie, IMC, tension, âge, etc.),
comparant régression logistique et random forest.

## Pourquoi ce projet

Jeu de données de référence en apprentissage supervisé (Pima Indians
Diabetes Dataset), utile pour pratiquer un pipeline de classification
médicale complet : détection d'un problème de qualité de données caché,
imputation sans fuite, gestion du déséquilibre de classes, comparaison de
modèles et interprétation des facteurs de risque.

## Jeu de données

`data/diabetes.csv` : 768 patientes, 8 variables cliniques (nombre de
grossesses, glycémie, tension artérielle, épaisseur du pli cutané,
insulinémie, IMC, fonction de pedigree du diabète, âge) et une variable
cible binaire `Outcome` (diabète ou non). ~34.9% de cas positifs.

## Point technique important : les zéros ne sont pas des zéros

**Ce jeu de données n'était accompagné d'aucun code** — et il contient un
piège classique et bien documenté sur ce dataset précis : les colonnes
`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` et `BMI` utilisent
**0 comme encodage de valeur manquante**, pas comme une vraie mesure (une
glycémie ou une tension artérielle de 0 chez une personne vivante n'existe
pas). Sans traitement, un modèle entraîné directement sur les données
brutes apprendrait des zéros artificiels comme s'il s'agissait de vraies
valeurs basses — un biais silencieux, sans erreur ni warning pour le
signaler.

Répartition réelle des zéros suspects dans le dataset :

| Colonne | % de zéros |
|---|---|
| Insulin | 48.7% |
| SkinThickness | 29.6% |
| BloodPressure | 4.6% |
| BMI | 1.4% |
| Glucose | 0.7% |

`src/data_prep.py` remplace ces zéros par de vraies valeurs manquantes
(NaN), puis les impute par la **médiane calculée uniquement sur le jeu
d'entraînement** (pas de fuite de données vers le test) — cette médiane
est aussi sauvegardée dans le modèle pour être réutilisée à l'identique
sur de nouveaux patients (`src/predict.py`).

## Structure du projet

```
diabetes-risk-prediction/
├── data/
│   ├── diabetes.csv
│   └── new_patients_example.csv   # exemple pour predict.py
├── src/
│   ├── data_prep.py                # correction zéros->NaN, split, imputation, scaling
│   ├── train.py                    # entraînement + comparaison de modèles
│   └── predict.py                  # prédiction sur de nouveaux patients
├── models/                         # modèle entraîné (model.pkl, généré)
├── outputs/                        # métriques générées (metrics.json)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
python3 src/train.py --data data/diabetes.csv
python3 src/predict.py --patients data/new_patients_example.csv
```

## Résultats obtenus (exécution réelle sur les 768 patientes)

| Modèle | Accuracy | Recall (classe 1) | ROC AUC |
|---|---|---|---|
| Régression logistique | 0.73 | 0.70 | 0.812 |
| Random Forest | 0.75 | 0.57 | **0.816** |

Random Forest légèrement meilleur en AUC, mais la régression logistique
détecte mieux les cas positifs (rappel 0.70 vs 0.57) — un compromis
important en contexte médical, où manquer un cas de diabète (faux négatif)
est en général plus coûteux qu'une fausse alerte. Les deux modèles
utilisent `class_weight="balanced"` pour compenser le déséquilibre des
classes (34.9% de cas positifs).

Facteurs de risque les plus déterminants (coefficients de la régression
logistique, sur variables standardisées) : **Glucose** (1.18), **BMI**
(0.71), puis Pregnancies (0.37) — cohérent avec la littérature médicale
sur le diabète de type 2.

## Stack technique

Python, pandas, numpy, scikit-learn (LogisticRegression, RandomForest,
StandardScaler).

## Licence

MIT — voir [LICENSE](./LICENSE).
