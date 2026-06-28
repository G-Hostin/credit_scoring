---
title: Credit Scoring API
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Credit Scoring - Modèle de scoring crédit "Prêt à dépenser"

Outil de scoring crédit pour la société (fictive) **Prêt à dépenser** : à partir des
informations d'un client, le modèle prédit la **probabilité de défaut de paiement** et
classe la demande en **accordée / refusée**.

Ce dépôt couvre deux projets de la formation OpenClassrooms *AI Engineer* :

- **Projet 6 - Initiez-vous au MLOps (partie 1/2)** : exploration des données, feature
  engineering, modélisation et suivi des expérimentations avec MLflow.
- **Projet 8 - Déployez et monitorez votre modèle (partie 2/2)** : mise en production du
  modèle via une API, conteneurisation Docker, CI/CD, stockage des données de production
  et monitoring (data drift).

## Démo en ligne

- **API** : https://credit-scoring-api-mn5w.onrender.com (documentation interactive sur `/docs`)
- **Dashboard de monitoring** : https://credit-scoring-dashboard-fndm.onrender.com

*(Services gratuits Render : le premier accès peut prendre ~1 min, le temps que le service se réveille.)*

## Architecture

```
                 POST /predict
   client  ───────────────────▶  API (FastAPI)  ──enregistre──▶  PostgreSQL
                                                                      │
                          Dashboard (Streamlit)  ◀──────lit──────────┘
```

L'API sert le modèle et **enregistre chaque prédiction** (entrées, sortie, latence) en
base. Le dashboard lit cette base pour suivre la **dérive des données** et la **santé
opérationnelle**. En local, les services tournent via `docker compose` ; en production,
ils sont déployés sur **Render** (API + base + dashboard).

## Modèle

- **Algorithme** : LightGBM (Pipeline scikit-learn : imputation médiane + classifieur).
- **Gestion du déséquilibre** : `scale_pos_weight ≈ 11.39` (≈ 92 % bons clients / 8 % défauts).
- **Métrique métier** : coût asymétrique `FN x 10 + FP x 1` (un faux négatif, défaut non
  détecté, coûte 10x plus cher qu'un faux positif).
- **Seuil de décision optimal** : **0.53** (voir `models/threshold.json`).
- Le modèle est entraîné et versionné avec MLflow (`models/credit_scoring_model/`). Pour la
  production, il est converti au format **ONNX** (`models/credit_scoring_model.onnx`) pour une
  inférence rapide (voir la section Optimisation).

## API

L'API (FastAPI) reçoit le dossier de features d'un client et retourne la probabilité de
défaut et la décision.

- `GET /health` : état de l'API et chargement du modèle.
- `POST /predict` : prédiction. Corps attendu : `{ "features": { "<nom_feature>": valeur, ... } }`
  avec les **795 features** du modèle (les valeurs peuvent être `null` : le modèle gère
  l'imputation). Réponse : `{ "probability_default", "decision", "threshold" }`.
- `GET /docs` : documentation interactive (Swagger).

## Lancer en local

Tout le stack (API + base PostgreSQL) avec Docker :

```bash
docker compose up --build
```

- API : http://localhost:8000 (`/docs` pour Swagger)
- Base PostgreSQL : `localhost:5433` (utilisateur `credit`, base `credit_scoring`)

Pour **remplir la base** avec du trafic simulé (un lot normal + un lot "dérivé" qui imite
une récession), place le dataset préparé dans `data/dataset.parquet` puis :

```bash
uv run python scripts/simulate_traffic.py --url http://localhost:8000 --normal 200 --drift 200
```

Pour lancer le **dashboard** en local :

```bash
uv run streamlit run monitoring/dashboard.py
```

## Monitoring (interprétation)

Le dashboard (et le notebook `notebooks/05_monitoring.ipynb`) comparent les données de
**production** à une **référence** (échantillon des données d'entraînement,
`monitoring/reference_sample.parquet`).

- **Dérive des données** : pour chaque feature surveillée, un **test de Kolmogorov-Smirnov**
  donne un **p-value**. Un **p-value < 0.05** signale une **dérive** (la distribution a
  réellement changé par rapport à l'entraînement). Le dashboard liste les features dérivées
  et superpose les distributions référence vs production.
- **Métriques opérationnelles** : nombre d'appels, **taux d'erreur**, **latence**
  (moyenne et p95), distribution des scores prédits.

Une dérive confirmée est un signal d'alerte : en situation réelle, elle déclencherait
une investigation et, si elle persiste, un **réentraînement** du modèle.

## Optimisation des performances

Le notebook `notebooks/06_optimization.ipynb` documente l'analyse de performance et les
optimisations (profiling avec `cProfile`, puis mesures avant / après) :

- **Inférence ONNX** : le modèle est converti en ONNX (`scripts/convert_to_onnx.py`) et servi
  via ONNX Runtime. Inférence environ 130x plus rapide, et image Docker allégée (plus de
  mlflow / lightgbm / scikit-learn / pandas au runtime : ~1,1 GB vers ~480 MB).
- **Écriture non-bloquante** : l'enregistrement en base se fait en tâche de fond
  (`BackgroundTasks`), ce qui retire ~47 ms du temps de réponse.
- **Sans régression** : les prédictions ONNX sont identiques à celles du modèle d'origine
  (écart de l'ordre de 1e-7).

## Déploiement et CI/CD

- **Déploiement** : décrit en infrastructure-as-code dans `render.yaml` (blueprint Render)
  qui crée 3 ressources : la base PostgreSQL, l'API (`Dockerfile`) et le dashboard
  (`Dockerfile.dashboard`).
- **CI** (`.github/workflows/ci.yml`) : sur chaque pull request, lance les tests (pytest)
  puis construit les images Docker.
- **CD** (`.github/workflows/cd.yml`) : sur un push vers `main`, relance tests + build puis
  **déclenche le déploiement sur Render** (via deploy hooks). Les secrets (token, hooks)
  sont gérés dans GitHub, jamais dans le code.

## Structure du dépôt

```
.
├── api/                 # API d'inférence FastAPI + accès base de données
├── models/              # Modèle MLflow + modèle ONNX + médianes + seuil de décision
├── monitoring/          # Dashboard Streamlit, référence de drift, notebook d'analyse
├── notebooks/           # Analyse P6 (01-04) + monitoring (05) + optimisation (06)
├── scripts/             # Simulation de trafic + conversion ONNX
├── tests/               # Tests unitaires de l'API
├── data/                # Données (non versionnées, voir .gitignore)
├── .github/workflows/   # Pipelines CI et CD
├── Dockerfile           # Image de l'API
├── Dockerfile.dashboard # Image du dashboard
├── docker-compose.yml   # Stack local (API + PostgreSQL)
├── render.yaml          # Blueprint de déploiement Render
├── pyproject.toml       # Dépendances (gérées avec uv)
└── README.md
```

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/). Le **cœur** (runtime de l'API) est
volontairement léger (FastAPI, ONNX Runtime, accès base). Le reste est rangé en groupes :
`modeling` (modèle d'origine + conversion ONNX), `monitoring` (dashboard), `notebooks`
(analyse P6) et `dev` (tests).

```bash
uv sync                                                        # cœur + dev (lancer / tester l'API)
uv sync --group notebooks --group modeling --group monitoring  # tout (notebooks, analyses)
uv run jupyter lab                                             # ouvrir les notebooks
```

## Données

Jeu de données [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
Les fichiers de données (`data/`) et les artefacts de suivi MLflow (`mlruns/`) ne sont pas
versionnés (volumineux). Le notebook `02_preparation.ipynb` produit le jeu de données
préparé à partir des CSV bruts.
