"""API de scoring crédit - Prêt à dépenser.

Sert le modèle de scoring (LightGBM du projet 6) converti au format **ONNX** pour une
inférence rapide. L'API reçoit le dossier de features d'un client et renvoie la
probabilité de défaut ainsi que la décision (accordé / refusé) selon le seuil métier (0.53).

Le modèle ONNX est chargé UNE seule fois au démarrage (voir `lifespan`). L'imputation des
valeurs manquantes (médianes) est appliquée en amont, en numpy. Chaque prédiction est
enregistrée en base en tâche de fond (si une base est configurée).
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.database import init_db, save_prediction

# --- Chemins (relatifs à ce fichier, pour fonctionner aussi dans Docker) ---
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ONNX_PATH = MODELS_DIR / "credit_scoring_model.onnx"
MEDIANS_PATH = MODELS_DIR / "imputer_medians.npy"
FEATURES_PATH = MODELS_DIR / "feature_names.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

# Espace mémoire rempli au démarrage (le modèle n'est jamais rechargé par requête).
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement des artefacts au démarrage de l'API, libération à l'arrêt."""
    session = ort.InferenceSession(str(ONNX_PATH))
    state["session"] = session
    state["input_name"] = session.get_inputs()[0].name
    state["medians"] = np.load(MEDIANS_PATH)  # médiane par feature (imputation des NaN)
    state["features"] = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
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
    version="2.0.0",
    lifespan=lifespan,
)


# --- Schémas d'entrée / sortie (documentés automatiquement dans Swagger) ---
class PredictionRequest(BaseModel):
    """Dossier d'un client : les 795 features attendues par le modèle.

    Les valeurs peuvent être `null` (données manquantes) : elles sont imputées par la
    médiane. En revanche, toutes les clés doivent être présentes.
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
        "model_loaded": "session" in state,
        "n_features": len(state.get("features", [])),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Calcule la probabilité de défaut et la décision pour un client.

    Chaque appel (succès comme erreur) est enregistré en base en tâche de fond
    (BackgroundTasks) : la réponse est renvoyée immédiatement, l'écriture en base
    n'allonge pas le temps de réponse. Sert au suivi de production (drift, erreurs, latence).
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

        # 3) Inférence : vecteur numpy -> imputation (médianes) -> ONNX Runtime.
        try:
            row = np.array(
                [[provided[f] if provided[f] is not None else np.nan for f in expected]],
                dtype=np.float32,
            )
            row = np.where(np.isnan(row), state["medians"], row)
            outputs = state["session"].run(None, {state["input_name"]: row})
            probability_default = float(outputs[1][0][1])  # P(classe 1 = défaut)
        except Exception as exc:  # garde-fou : toute erreur inattendue -> 500 explicite
            raise HTTPException(status_code=500, detail=f"Erreur lors de la prediction : {exc}")

    except HTTPException as exc:
        # On enregistre l'échec en tâche de fond (pour le taux d'erreur) puis on relaie.
        background_tasks.add_task(
            save_prediction,
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

    # On enregistre la prédiction réussie en tâche de fond (après la réponse).
    background_tasks.add_task(
        save_prediction,
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
