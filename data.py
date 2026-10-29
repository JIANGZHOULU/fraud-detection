"""Synthetic card transactions (inspired by public credit-card fraud datasets)."""
import numpy as np
import pandas as pd

FEATURES = ["amount", "hour", "distance_from_home_km", "merchant_risk", "txns_last_24h", "is_online"]
TARGET = "is_fraud"


def make_data(n_rows: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(3.8, 1.0, n_rows), 1, 5000)          # transaction value
    hour = rng.integers(0, 24, n_rows)                                   # hour of day
    distance = np.clip(rng.exponential(30, n_rows), 0, 3000)            # km from the card holder's home
    merchant_risk = rng.beta(2, 5, n_rows)                               # 0 = trusted merchant, 1 = risky
    txns = rng.poisson(3, n_rows)                                        # card activity in the last 24 h
    online = (rng.random(n_rows) < 0.4).astype(int)

    night = (hour < 5)
    score = (
        -6.0
        + 7.0 * (night & (online == 1) & (amount > 120))                 # online spending at night
        + 7.0 * ((distance > 60) & (amount > 150))                       # large purchase far from home
        + 7.0 * (txns > 7)                                               # burst of activity
        + 7.0 * (merchant_risk > 0.7)                                    # risky merchant
    )
    fraud = rng.random(n_rows) < 1 / (1 + np.exp(-score))
    return pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "distance_from_home_km": distance.round(1),
        "merchant_risk": merchant_risk.round(3),
        "txns_last_24h": txns,
        "is_online": online,
        TARGET: fraud.astype(int),
    })
