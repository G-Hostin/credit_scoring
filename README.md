# Credit Scoring - Modèle de scoring crédit "Prêt à dépenser"

Outil de scoring crédit pour la société (fictive) **Prêt à dépenser** : à partir des
informations d'un client, le modèle prédit la **probabilité de défaut de paiement** et
classe la demande en **accordée / refusée**.

Ce dépôt couvre deux projets de la formation OpenClassrooms *AI Engineer* :

- **Projet 6 - Initiez-vous au MLOps (partie 1/2)** : exploration des données, feature
  engineering, modélisation et suivi des expérimentations avec MLflow.
- **Projet 8 - Déployez et monitorez votre modèle (partie 2/2)** : mise en production du
  modèle via une API, conteneurisation Docker, CI/CD, stockage des données de production
  et monitoring (data drift). *(en cours)*

## Modèle

- **Algorithme** : LightGBM (Pipeline scikit-learn : imputation médiane + classifieur).
- **Gestion du déséquilibre** : `scale_pos_weight ≈ 11.39` (≈ 92 % bons clients / 8 % défauts).
- **Métrique métier** : coût asymétrique `FN × 10 + FP × 1` (un faux négatif - défaut non
  détecté - coûte 10× plus cher qu'un faux positif).
- **Seuil de décision optimal** : **0.53** (voir `models/threshold.json`).
- Le modèle est packagé au format MLflow dans `models/credit_scoring_model/`.

## Structure du dépôt

```
.
├── notebooks/      # Analyse P6 : 01 EDA, 02 préparation, 03 modélisation, 04 optimisation
├── models/         # Modèle entraîné (MLflow) + seuil de décision
├── api/            # API d'inférence (projet 8)
├── tests/          # Tests unitaires (projet 8)
├── monitoring/     # Dashboard / analyse de data drift (projet 8)
├── data/           # Données (non versionnées - voir .gitignore)
├── pyproject.toml  # Dépendances (gérées avec uv)
└── README.md
```

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

## Données

Jeu de données [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
Les fichiers de données (`data/`) et les artefacts de suivi MLflow (`mlruns/`) ne sont pas
versionnés (volumineux). Le notebook `02_preparation.ipynb` produit le jeu de données
préparé à partir des CSV bruts.
