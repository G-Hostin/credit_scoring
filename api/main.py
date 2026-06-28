"""API de scoring crédit - Prêt à dépenser.

Expose le modèle de scoring (Pipeline scikit-learn : imputation médiane + LightGBM)
entraîné au projet 6. L'API reçoit le dossier de features d'un client et renvoie la
probabilité de défaut ainsi que la décision (accordé / refusé) selon le seuil métier
optimal (0.53).

Le modèle est chargé UNE seule fois au démarrage (voir `lifespan`), puis réutilisé
pour toutes les requêtes. Chaque prédiction est enregistrée en base (si configurée).
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.database import init_db, save_prediction

# --- Chemins (relatifs à ce fichier, pour fonctionner aussi dans Docker) ---
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "credit_scoring_model"
THRESHOLD_PATH = ROOT_DIR / "models" / "threshold.json"

# Espace mémoire rempli au démarrage (le modèle n'est jamais rechargé par requête).
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement des artefacts au démarrage de l'API, libération à l'arrêt."""
    state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    # Source de vérité : les noms et l'ordre des features vus à l'entraînement.
    state["features"] = list(state["model"].feature_names_in_)
    state["threshold"] = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"]
    init_db()  # crée la table 'predictions' si une base est configurée
    yield
    state.clear()


app = FastAPI(
    title="Credit Scoring API - Prêt à dépenser",
    description=(
        "API de scoring crédit. Reçoit le dossier de features d'un client "
        "et retourne la probabilité de défaut et la décision (accordé / refusé)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Schémas d'entrée / sortie (documentés automatiquement dans Swagger) ---
class PredictionRequest(BaseModel):
    """Dossier d'un client : les 795 features attendues par le modèle.

    Les valeurs peuvent être `null` (données manquantes) : le modèle gère
    l'imputation en interne. En revanche, toutes les clés doivent être présentes.
    """

    features: dict[str, float | None] = Field(
        ..., description="Dictionnaire {nom_feature: valeur} des 795 features du client."
    )


class PredictionResponse(BaseModel):
    probability_default: float = Field(..., description="Probabilité de défaut (classe 1), entre 0 et 1.")
    decision: str = Field(..., description="'accordé' ou 'refusé' selon le seuil métier.")
    threshold: float = Field(..., description="Seuil de décision appliqué.")


# --- Validation métier (quelques règles d'exemple, facilement extensibles) ---
def validate_business_rules(features: dict[str, float | None]) -> list[str]:
    """Retourne la liste des erreurs métier (vide si tout est valide)."""
    errors: list[str] = []

    income = features.get("AMT_INCOME_TOTAL")
    if income is not None and income <= 0:
        errors.append("AMT_INCOME_TOTAL doit etre strictement positif.")

    credit = features.get("AMT_CREDIT")
    if credit is not None and credit <= 0:
        errors.append("AMT_CREDIT doit etre strictement positif.")

    # DAYS_BIRTH est exprimé en jours négatifs (ex. -12000). Une valeur positive
    # ou un âge irréaliste (> 120 ans) est une erreur.
    days_birth = features.get("DAYS_BIRTH")
    if days_birth is not None and not (-43800 <= days_birth <= 0):
        errors.append("DAYS_BIRTH doit etre negatif et correspondre a un age plausible (<= 120 ans).")

    return errors


# --- Endpoints ---
@app.get("/", tags=["général"])
def root():
    """Point d'entrée simple."""
    return {"message": "Credit Scoring API - voir /docs pour la documentation interactive."}


@app.get("/health", tags=["général"])
def health():
    """Vérifie que l'API tourne et que le modèle est bien chargé (utile au déploiement)."""
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "n_features": len(state.get("features", [])),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(request: PredictionRequest):
    """Calcule la probabilité de défaut et la décision pour un client.

    Chaque appel (succès comme erreur) est enregistré en base avec sa latence,
    pour permettre le suivi de production (drift, taux d'erreur, temps d'inférence).
    """
    start = time.perf_counter()
    expected = state["features"]
    provided = request.features

    try:
        # 1) Toutes les features attendues sont-elles présentes ?
        missing = [f for f in expected if f not in provided]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"{len(missing)} feature(s) manquante(s).",
                    "missing_features": missing[:20],  # on n'en montre qu'un échantillon
                },
            )

        # 2) Règles métier (valeurs hors plage)
        business_errors = validate_business_rules(provided)
        if business_errors:
            raise HTTPException(
                status_code=422,
                detail={"message": "Validation metier echouee.", "errors": business_errors},
            )

        # 3) Construction de la ligne dans le bon ordre, puis prédiction
        try:
            row = {feature: provided[feature] for feature in expected}
            X = pd.DataFrame([row], columns=expected).astype(float)
            probability_default = float(state["model"].predict_proba(X)[0, 1])
        except Exception as exc:  # garde-fou : toute erreur inattendue -> 500 explicite
            raise HTTPException(status_code=500, detail=f"Erreur lors de la prediction : {exc}")

    except HTTPException as exc:
        # On enregistre l'échec (pour le taux d'erreur) puis on relaie l'erreur.
        save_prediction(
            features=provided,
            probability_default=None,
            decision=None,
            threshold=state["threshold"],
            latency_ms=(time.perf_counter() - start) * 1000,
            status="error",
            error_message=str(exc.detail),
        )
        raise

    threshold = state["threshold"]
    decision = "refusé" if probability_default >= threshold else "accordé"
    latency_ms = (time.perf_counter() - start) * 1000

    # On enregistre la prédiction réussie.
    save_prediction(
        features=provided,
        probability_default=probability_default,
        decision=decision,
        threshold=threshold,
        latency_ms=latency_ms,
        status="success",
        error_message=None,
    )

    return PredictionResponse(
        probability_default=probability_default,
        decision=decision,
        threshold=threshold,
    )
