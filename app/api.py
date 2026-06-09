import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── JSON-логгер ──────────────────────────────────────────────────────────────
os.makedirs("app", exist_ok=True)
log_handler = logging.FileHandler("app/api.log", encoding="utf-8")
log_handler.setFormatter(logging.Formatter("%(message)s"))
api_logger = logging.getLogger("api")
api_logger.setLevel(logging.INFO)
api_logger.addHandler(log_handler)


def log_request(user_id, model_version, prediction, probability):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "model_version": model_version,
        "prediction": prediction,
        "probability": probability,
    }
    api_logger.info(json.dumps(record, ensure_ascii=False))


# ── Загрузка моделей ─────────────────────────────────────────────────────────
MODEL_V1_PATH = "models/model_v1.pkl"
MODEL_V2_PATH = "models/model_v2.pkl"

model_v1 = joblib.load(MODEL_V1_PATH)
model_v2_bundle = joblib.load(MODEL_V2_PATH)   # {"model": lr, "scaler": scaler}

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


# ── A/B роутинг ───────────────────────────────────────────────────────────────
def get_model_version(user_id) -> str:
    hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    return "v2" if hash_val % 2 == 1 else "v1"


def select_model(user_id: str | None):
    """Возвращает (версия, предсказатель).
    Если user_id не передан — всегда v1 (детерминированный fallback).
    """
    if user_id is None:
        return "v1", _predict_v1

    version = get_model_version(user_id)
    return (version, _predict_v1) if version == "v1" else (version, _predict_v2)


def _predict_v1(X: np.ndarray):
    prediction = int(model_v1.predict(X)[0])
    probability = float(model_v1.predict_proba(X)[0][1])
    return prediction, probability


def _predict_v2(X: np.ndarray):
    scaler = model_v2_bundle["scaler"]
    model  = model_v2_bundle["model"]
    X_sc   = scaler.transform(X)
    prediction = int(model.predict(X_sc)[0])
    probability = float(model.predict_proba(X_sc)[0][1])
    return prediction, probability


# ── Эндпоинты ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(force=True)

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    user_id = data.get("user_id")           # опциональный параметр
    model_version, predictor = select_model(str(user_id) if user_id is not None else None)

    X = np.array([[data[f] for f in FEATURES]])
    prediction, probability = predictor(X)
    probability = round(probability, 4)

    log_request(user_id, model_version, prediction, probability)

    return jsonify({
        "prediction":    prediction,
        "probability":   probability,
        "model_version": model_version,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
