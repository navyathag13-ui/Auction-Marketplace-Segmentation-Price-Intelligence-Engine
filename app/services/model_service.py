"""
Model inference service
Loads the trained LightGBM model and provides price scoring
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

log = logging.getLogger(__name__)

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_LABELS = {0: "underpriced", 1: "fair_priced", 2: "overpriced"}


@lru_cache(maxsize=1)
def load_model():
    model_path   = os.path.join(MODELS_DIR, "lgbm_price_classifier.joblib")
    features_path = os.path.join(MODELS_DIR, "model_features.json")

    if not os.path.exists(model_path):
        log.warning("LightGBM model not found — price scoring unavailable")
        return None, []

    model = joblib.load(model_path)
    features = []
    if os.path.exists(features_path):
        with open(features_path) as f:
            features = json.load(f)
    log.info(f"Loaded LightGBM model ({len(features)} features)")
    return model, features


@lru_cache(maxsize=1)
def load_feature_importances() -> list[dict]:
    model, features = load_model()
    if model is None or not features:
        return []
    importances = model.feature_importances_
    fi = sorted(
        [{"feature": f, "importance": round(float(i), 4)}
         for f, i in zip(features, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    for rank, item in enumerate(fi, 1):
        item["rank"] = rank
    return fi


def predict_price_label(feature_row: dict) -> dict:
    """
    Given a dict of feature values, return price classification and probabilities.
    Missing features are filled with 0.
    """
    model, features = load_model()
    if model is None:
        return {"error": "Model not loaded"}

    vals = np.array([[feature_row.get(f, 0.0) for f in features]])
    proba = model.predict_proba(vals)[0]
    pred  = int(np.argmax(proba))

    return {
        "price_label":     CLASS_LABELS[pred],
        "price_label_enc": pred,
        "prob_underpriced": round(float(proba[0]), 4),
        "prob_fair_priced": round(float(proba[1]), 4),
        "prob_overpriced":  round(float(proba[2]), 4),
        "price_score":      round(float(proba[1]) * 100, 1),
    }
