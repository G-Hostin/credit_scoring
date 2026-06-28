"""Simulation de trafic de production pour alimenter la base de prédictions.

Faute de vrais utilisateurs, on rejoue de vrais dossiers clients à travers l'API :
- un lot "normal" (clients tirés du dataset, distribution proche de l'entraînement),
- un lot "dérivé" simulant une récession (revenus en baisse, population plus jeune),
  pour démontrer ensuite que le monitoring détecte bien la dérive (data drift).

Chaque appel est enregistré en base par l'API ; ces données serviront à l'analyse.

Usage :
    python scripts/simulate_traffic.py --url http://localhost:8000 --normal 200 --drift 200
"""

import argparse

import httpx
import pandas as pd


def clean_columns(columns):
    """Même nettoyage des noms de colonnes qu'à l'entraînement (notebook 03)."""
    return [c.replace(" ", "_").replace(",", "_").replace(":", "_") for c in columns]


def row_to_payload(row, features):
    """Transforme une ligne du dataset en corps de requête {features: {...}}."""
    return {f: (None if pd.isna(row[f]) else float(row[f])) for f in features}


def apply_recession_drift(row):
    """Simule un scénario de récession : revenus -40 %, population plus jeune."""
    row = row.copy()
    if pd.notna(row.get("AMT_INCOME_TOTAL")):
        row["AMT_INCOME_TOTAL"] *= 0.6
    if pd.notna(row.get("DAYS_BIRTH")):
        row["DAYS_BIRTH"] *= 0.7  # valeur négative -> moins négative -> plus jeune
    return row


def send_batch(client, url, rows, features, label):
    """Envoie un lot de dossiers à l'API et compte les prédictions réussies."""
    success = 0
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        payload = {"features": row_to_payload(row, features)}
        try:
            response = client.post(f"{url}/predict", json=payload, timeout=30)
            if response.status_code == 200:
                success += 1
        except Exception:
            pass
        if i % 50 == 0:
            print(f"  {label}: {i}/{len(rows)} envoyés")
    print(f"  {label}: {success}/{len(rows)} prédictions réussies")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="URL de base de l'API")
    parser.add_argument("--data", default="data/dataset.parquet", help="Chemin du dataset préparé")
    parser.add_argument("--normal", type=int, default=200, help="Nombre de clients du lot normal")
    parser.add_argument("--drift", type=int, default=200, help="Nombre de clients du lot dérivé")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df.columns = clean_columns(df.columns)
    features = [c for c in df.columns if c not in ("TARGET", "SK_ID_CURR")]

    normal_rows = df.sample(args.normal, random_state=args.seed)
    drift_rows = df.sample(args.drift, random_state=args.seed + 1).apply(apply_recession_drift, axis=1)

    with httpx.Client() as client:
        print("Lot normal :")
        send_batch(client, args.url, normal_rows, features, "normal")
        print("Lot dérivé (récession) :")
        send_batch(client, args.url, drift_rows, features, "drift")

    print("Terminé.")


if __name__ == "__main__":
    main()
