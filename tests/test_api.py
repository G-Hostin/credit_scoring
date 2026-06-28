"""Tests automatisés de l'API de scoring crédit.

Couvre les cas nominaux (clients aux profils contrastés) et les cas critiques
demandés par le brief : feature manquante, mauvais type, valeur hors plage.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

# Jeu de données de test : 2 clients réels aux profils opposés.
_FIXTURE = json.loads((Path(__file__).parent / "sample_clients.json").read_text(encoding="utf-8"))
GOOD = _FIXTURE["good_client"]["features"]
BAD = _FIXTURE["bad_client"]["features"]


@pytest.fixture(scope="module")
def client():
    """Client de test ; le bloc `with` déclenche le lifespan (modèle chargé une fois)."""
    with TestClient(app) as test_client:
        yield test_client


# --- Cas nominaux ---
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 795


def test_good_client_is_accepted(client):
    response = client.post("/predict", json={"features": GOOD})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "accordé"
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["threshold"] == 0.53


def test_bad_client_is_rejected(client):
    response = client.post("/predict", json={"features": BAD})
    assert response.status_code == 200
    assert response.json()["decision"] == "refusé"


def test_good_client_is_less_risky_than_bad(client):
    proba_good = client.post("/predict", json={"features": GOOD}).json()["probability_default"]
    proba_bad = client.post("/predict", json={"features": BAD}).json()["probability_default"]
    assert proba_good < proba_bad


# --- Cas critiques (validation des entrées) ---
def test_missing_feature_returns_422(client):
    payload = dict(GOOD)
    payload.pop(next(iter(payload)))  # on retire une feature obligatoire
    response = client.post("/predict", json={"features": payload})
    assert response.status_code == 422


def test_wrong_type_returns_422(client):
    payload = dict(GOOD)
    payload["AMT_CREDIT"] = "texte_invalide"  # texte là où un nombre est attendu
    response = client.post("/predict", json={"features": payload})
    assert response.status_code == 422


def test_out_of_range_income_returns_422(client):
    payload = dict(GOOD)
    payload["AMT_INCOME_TOTAL"] = 0  # revenu nul : interdit par la règle métier
    response = client.post("/predict", json={"features": payload})
    assert response.status_code == 422
