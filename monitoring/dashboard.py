"""Dashboard de monitoring (Streamlit) du modèle de scoring crédit.

Affiche, à partir de la référence (données d'entraînement) et des données de
production (PostgreSQL) :
- des métriques opérationnelles (appels, taux d'erreur, latence, scores) ;
- la dérive des données (Evidently), avec les distributions des features dérivées
  et le rapport Evidently complet.

Lancement local : streamlit run monitoring/dashboard.py
"""

import os
import tempfile

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import create_engine

from evidently import Report
from evidently.presets import DataDriftPreset

px.defaults.template = "plotly_dark"  # graphiques en thème sombre

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference_sample.parquet")

# Base de production : variable d'environnement, sinon la base locale (docker-compose).
DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://credit:credit@localhost:5433/credit_scoring"
)
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)

MONITORED = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "PAYMENT_RATE", "ANNUITY_INCOME_PERC",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]


@st.cache_data(ttl=60)
def load_data():
    """Charge la référence et les données de production (cache 60 s)."""
    reference = pd.read_parquet(REFERENCE_PATH)
    engine = create_engine(DB_URL)
    prod = pd.read_sql(
        "SELECT created_at, features, probability_default, latency_ms, status FROM predictions",
        engine,
    )
    return reference, prod


@st.cache_data(ttl=60)
def compute_drift(reference, production):
    """Calcule la dérive (Evidently) et renvoie le tableau + le rapport HTML."""
    snapshot = Report([DataDriftPreset()]).run(current_data=production, reference_data=reference)
    rows = []
    for metric in snapshot.dict()["metrics"]:
        if metric["metric_name"].startswith("ValueDrift"):
            rows.append({"feature": metric["config"]["column"], "p_value": metric["value"]})
    drift = pd.DataFrame(rows).sort_values("p_value")
    drift["dérive"] = drift["p_value"] < 0.05

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
        snapshot.save_html(tmp.name)
        report_html = open(tmp.name, encoding="utf-8").read()
    return drift, report_html


st.set_page_config(page_title="Monitoring - Credit Scoring", page_icon="💳", layout="wide")
st.title("💳 Monitoring du modèle de scoring crédit")
st.caption("Suivi de la dérive des données et des performances en production")

reference, prod = load_data()

if prod.empty:
    st.warning("Aucune donnée de production pour l'instant. Envoyez des prédictions à l'API.")
    st.stop()

production = pd.DataFrame(prod["features"].tolist())[MONITORED]

with st.sidebar:
    st.header("À propos")
    st.write(
        "Ce tableau de bord surveille le modèle de scoring crédit en production : "
        "dérive des données (par rapport à l'entraînement) et santé opérationnelle de l'API."
    )
    st.caption("Source : base PostgreSQL de production")

# --- Indicateurs clés ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Appels logués", len(prod))
k2.metric("Taux d'erreur", f"{(prod['status'] == 'error').mean():.1%}")
k3.metric("Latence moyenne", f"{prod['latency_ms'].mean():.0f} ms")
k4.metric("Latence p95", f"{prod['latency_ms'].quantile(0.95):.0f} ms")

tab_ops, tab_drift = st.tabs(["📈 Opérationnel", "🔀 Dérive des données"])

with tab_ops:
    st.plotly_chart(
        px.histogram(prod, x="probability_default", nbins=30,
                     title="Distribution des probabilités de défaut prédites"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.histogram(prod, x="latency_ms", nbins=30, title="Distribution de la latence (ms)"),
        use_container_width=True,
    )

with tab_drift:
    drift, report_html = compute_drift(reference, production)
    n_drift = int(drift["dérive"].sum())
    if n_drift:
        st.error(f"⚠️ Dérive détectée sur {n_drift} feature(s) (p-value < 0.05).")
    else:
        st.success("✅ Aucune dérive détectée.")
    st.dataframe(drift, use_container_width=True, hide_index=True)

    # Deux courbes de densité superposées (référence vs production) sur le même axe.
    for feature in drift.loc[drift["dérive"], "feature"]:
        ref_values = reference[feature].dropna().tolist()
        prod_values = production[feature].dropna().tolist()
        fig = ff.create_distplot(
            [ref_values, prod_values],
            group_labels=["référence", "production"],
            show_hist=False, show_rug=False,
        )
        fig.update_layout(template="plotly_dark", title=f"Distribution : {feature}")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Voir le rapport Evidently complet"):
        components.html(report_html, height=800, scrolling=True)
