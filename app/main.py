import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "prediction"):
            payload["prediction"] = record.prediction
        return json.dumps(payload)


logger = logging.getLogger("mp263.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.propagate = False


class PredictRequest(BaseModel):
    flu_cases_reported: float = Field(..., le=350)
    temp_c: float
    nearby_clinics: float = Field(..., ge=0)
    discount_pct: float
    stock_lines: float
    distance_to_hospital_km: float
    shop_type: str
    locality: str


MODEL_PATH = Path("models/model.joblib")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


app = FastAPI(title="MP263 Pharmacy Daily Sales API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    model = get_model()
    features = pd.DataFrame([payload.model_dump()])
    pred = float(model.predict(features)[0])
    logger.info("prediction_served", extra={"prediction": round(pred, 4)})
    return {"prediction": pred}
