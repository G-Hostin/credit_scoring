"""Convertit le modèle de scoring (LightGBM du projet 6) au format ONNX.

L'API charge ensuite ce modèle ONNX (inférence rapide via ONNX Runtime) au lieu du
Pipeline scikit-learn/MLflow. L'imputation des valeurs manquantes est faite en amont
dans le code de l'API, à partir des médianes extraites ici.

Produit dans `models/` :
- `credit_scoring_model.onnx` : le modèle ONNX (LightGBM),
- `imputer_medians.npy`       : la médiane de chaque feature (imputation des NaN),
- `feature_names.json`        : la liste ordonnée des 795 features attendues.

Usage : uv run python scripts/convert_to_onnx.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import mlflow.sklearn
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Modèle d'origine (Pipeline : imputation médiane + LightGBM).
model = mlflow.sklearn.load_model(str(MODELS_DIR / "credit_scoring_model"))
features = list(model.feature_names_in_)
medians = model.named_steps["imputer"].statistics_.astype(np.float32)
lgbm = model.named_steps["model"]

# Conversion du LightGBM en ONNX (entrée = vecteur des features).
onnx_model = convert_lightgbm(
    lgbm,
    initial_types=[("input", FloatTensorType([None, len(features)]))],
    zipmap=False,
)

(MODELS_DIR / "credit_scoring_model.onnx").write_bytes(onnx_model.SerializeToString())
np.save(MODELS_DIR / "imputer_medians.npy", medians)
(MODELS_DIR / "feature_names.json").write_text(json.dumps(features), encoding="utf-8")

print(f"OK : {len(features)} features. Artefacts ONNX écrits dans models/.")
