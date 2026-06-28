# Image de base légère avec Python 3.13 (même version que le projet)
FROM python:3.13-slim

# LightGBM a besoin de la librairie OpenMP (libgomp) au moment de l'exécution.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# On récupère "uv" depuis son image officielle (même gestionnaire qu'en local).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 1) Les dépendances d'abord : Docker met cette couche en cache tant que
#    pyproject.toml / uv.lock ne changent pas (rebuilds plus rapides).
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# 2) Puis le code de l'API et le modèle.
COPY api ./api
COPY models ./models

# Le venv créé par uv devient le Python par défaut ; /app est importable.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Hugging Face Spaces impose le port 7860.
EXPOSE 7860

# Démarrage : le modèle est chargé une seule fois (lifespan), puis l'API répond.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
