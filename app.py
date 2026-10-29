"""FastAPI service for Credit Card Fraud Detection."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ['amount', 'hour', 'distance_from_home_km', 'merchant_risk', 'txns_last_24h', 'is_online']

# TODO 3d: use the same THRESHOLD you chose in train.py (TODO 2a). Replace "____" with that number, e.g. "0.3".
# THINK: two copies of one number can drift apart. How could the API read it from MLflow or metrics.json?
DEFAULT_THRESHOLD = "0.3"
if DEFAULT_THRESHOLD == "____":
    raise RuntimeError("TODO 3d in app.py: set DEFAULT_THRESHOLD to the threshold you chose in train.py")
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fraud-api")


def load_model():
    uri = os.getenv("MODEL_URI")                 # e.g. models:/fraud-model@production
    if uri:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri), uri
    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")


# TODO 3a: set limits (ge = minimum, le = maximum) so impossible values get a 422 error.
# One field is done for you as an example. Training data ranges (from df.describe() in Part 1):
#   amount                     1 - 1413.81
#   hour                       0 - 23
#   distance_from_home_km      0 - 281.5
#   merchant_risk              0 - 0.91
#   txns_last_24h              0 - 12
#   is_online                  0 - 1
# Options for how wide to make the limits:
#   Option A: exactly the training min/max  -> safest for the model, but rejects some real requests
#   Option B: real-world limits             -> e.g. 0-1 for yes/no fields, 0-23 for hours
# THINK: what should happen to a value that is possible but outside the training data? Reject? Warn?
# class Request(BaseModel):
#     amount: float = Field(..., ge=0, le=10000, description="Transaction amount ($)", examples=[600.0])
#     hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)", examples=[2])
#     distance_from_home_km: float = Field(..., ge=None, le=None, description="Distance from card holder's home (km)", examples=[120.0])
#     merchant_risk: float = Field(..., ge=None, le=None, description="Merchant risk score (0 = trusted, 1 = risky)", examples=[0.8])
#     txns_last_24h: int = Field(..., ge=None, le=None, description="Card transactions in the last 24 hours", examples=[9])
#     is_online: int = Field(..., ge=None, le=None, description="1 = online purchase, 0 = in store", examples=[1])

class Request(BaseModel):
    amount: float = Field(..., ge=0, le=10000, description="Transaction amount ($)", examples=[600.0])
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)", examples=[2])
    distance_from_home_km: float = Field(..., ge=0, le=20000, description="Distance from card holder's home (km)", examples=[120.0])
    merchant_risk: float = Field(..., ge=0.0, le=1.0, description="Merchant risk score (0 = trusted, 1 = risky)", examples=[0.8])
    txns_last_24h: int = Field(..., ge=0, le=200, description="Card transactions in the last 24 hours", examples=[9])
    is_online: int = Field(..., ge=0, le=1, description="1 = online purchase, 0 = in store", examples=[1])

class Prediction(BaseModel):
    probability: float
    flagged: bool
    recommended_action: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    # TODO 3c: return a dict with the keys "model_source", "threshold" and "features".
    # HINT: the values are MODEL_SOURCE, THRESHOLD and FEATURES (all defined above).
    # THINK: who would call this endpoint, and when? (someone debugging a strange prediction?)
    return {
        "model_source": MODEL_SOURCE,
        "threshold": THRESHOLD,
        "features": FEATURES
    }


@app.post("/predict", response_model=Prediction)
def predict(request: Request):
    start = time.perf_counter()
    X = pd.DataFrame([request.model_dump()])[FEATURES]
    probability = float(model.predict_proba(X)[0, 1])
    flagged = probability >= THRESHOLD

    # TODO 3b: turn the prediction into an action for the payment system and the bank's fraud team.
    #   Option A (2 levels): if flagged -> "Ask the customer to confirm with a one-time code", else "Approve the payment"
    #   Option B (3 levels): also "Block the card and call the customer" when probability >= 0.7
    #   Option C (your own): e.g. include the probability in the message
    # THINK: who reads this? What would make them trust (or ignore) it?
    # action = "TODO"
    if probability >= 0.7:
        action = "Block the card and call the customer"
    elif probability >= THRESHOLD:  # >= 0.3 but < 0.7
        action = "Ask the customer to confirm with a one-time code"
    else:                           # < 0.3
        action = "Approve the payment"

    log.info("predict p=%.3f flagged=%s latency_ms=%.1f", probability, flagged, (time.perf_counter() - start) * 1000)
    return Prediction(probability=round(probability, 4), flagged=flagged, recommended_action=action)
