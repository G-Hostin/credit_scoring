"""Stockage des prédictions de production dans PostgreSQL (via SQLAlchemy).

Le logging est OPTIONNEL : si la variable d'environnement `DATABASE_URL` n'est pas
définie (ex. déploiement HF sans base), l'API fonctionne normalement, sans rien
enregistrer. Une erreur d'écriture en base ne doit jamais casser la réponse de l'API.
"""

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger("api.database")

Base = declarative_base()


class Prediction(Base):
    """Une ligne = une prédiction servie par l'API (un appel à /predict)."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    features = Column(JSON)                 # les inputs (les 795 features du client)
    probability_default = Column(Float)     # output : probabilité de défaut
    decision = Column(String)               # output : "accordé" / "refusé"
    threshold = Column(Float)               # seuil de décision appliqué
    latency_ms = Column(Float)              # temps d'inférence (millisecondes)
    status = Column(String)                 # "success" ou "error"
    error_message = Column(String, nullable=True)  # détail si erreur


def _normalize_url(url: str) -> str:
    """Force l'usage du pilote psycopg v3 et corrige le préfixe `postgres://`.

    Les hébergeurs (Render, Neon...) donnent souvent une URL `postgres://` ou
    `postgresql://` ; SQLAlchemy choisirait alors le vieux pilote psycopg2.
    """
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# --- Connexion : créée uniquement si DATABASE_URL est définie ---
DATABASE_URL = os.environ.get("DATABASE_URL")
_engine = (
    create_engine(_normalize_url(DATABASE_URL), pool_pre_ping=True)
    if DATABASE_URL
    else None
)
_Session = sessionmaker(bind=_engine) if _engine else None


def init_db() -> None:
    """Crée la table `predictions` si besoin. Appelé au démarrage de l'API."""
    if _engine is None:
        logger.info("DATABASE_URL non definie : logging des predictions desactive.")
        return
    Base.metadata.create_all(_engine)
    logger.info("Base de donnees prete (table 'predictions').")


def save_prediction(**fields) -> None:
    """Enregistre une prédiction en base. Sans base configurée, ne fait rien.

    Toute erreur est seulement journalisée (jamais propagée à l'API).
    """
    if _Session is None:
        return
    try:
        with _Session() as session:
            session.add(Prediction(**fields))
            session.commit()
    except Exception as exc:  # un souci de log ne doit pas interrompre l'API
        logger.warning("Echec de l'enregistrement de la prediction : %s", exc)
