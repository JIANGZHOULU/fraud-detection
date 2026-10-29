import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
NORMAL = {"amount": 35.0, "hour": 14, "distance_from_home_km": 3.0, "merchant_risk": 0.1, "txns_last_24h": 2, "is_online": 0}
RISKY = {"amount": 600.0, "hour": 2, "distance_from_home_km": 120.0, "merchant_risk": 0.8, "txns_last_24h": 9, "is_online": 1}


# ---------- given tests ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_has_threshold():                            # checks TODO 3c
    body = client.get("/model-info").json()
    assert {"model_source", "threshold", "features"} <= set(body)


def test_predict_returns_valid_response():
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert isinstance(body["flagged"], bool)


def test_risky_case_scores_higher_than_normal():
    normal = client.post("/predict", json=NORMAL).json()
    risky = client.post("/predict", json=RISKY).json()
    assert risky["probability"] > normal["probability"]
    assert risky["flagged"] is True


def test_actions_are_filled_in_and_differ():                    # checks TODO 3b
    normal = client.post("/predict", json=NORMAL).json()["recommended_action"]
    risky = client.post("/predict", json=RISKY).json()["recommended_action"]
    assert "TODO" not in (normal + risky)
    assert normal != risky


# ---------- your tests ----------
def test_out_of_range_input_is_rejected():
    # TODO 4a: send a request with an impossible value and assert the API answers 422.
    # HINT: r = client.post("/predict", json={**NORMAL, "amount": -5}); assert r.status_code == 422
    # THINK: test both edges? (a value just below your minimum AND just above your maximum)
    # assert False, "TODO 4a: write this test"

    # Test below minimum (negative amount)
    r_below = client.post("/predict", json={**NORMAL, "amount": -5.0})
    assert r_below.status_code == 422, "Should reject negative amount"

    # Test above maximum (amount > 10000)
    r_above = client.post("/predict", json={**NORMAL, "amount": 15000.0})
    assert r_above.status_code == 422, "Should reject amount > 10000"

    # Test hour out of bounds
    r_hour = client.post("/predict", json={**NORMAL, "hour": 25})
    assert r_hour.status_code == 422, "Should reject hour > 23"


def test_missing_field_is_rejected():
    # TODO 4b: remove one field from NORMAL, send it, and assert 422.
    # HINT: request = dict(NORMAL); request.pop("is_online")
    # assert False, "TODO 4b: write this test"
    request = dict(NORMAL)
    request.pop("is_online")  # Remove a required field

    r = client.post("/predict", json=request)
    assert r.status_code == 422, "Should reject request missing a required field"


def test_my_own_idea():
    # TODO 4c (optional, recommended): write a test of your own. Ideas:
    #   - a higher merchant_risk never LOWERS the probability (a "behaviour" test)
    #   - a text value such as "amount": "high" is rejected with 422
    #   - /predict answers in under 200 ms
    #   - changing a value the model should ignore (e.g. sending it twice) gives the same answer
    # pytest.skip("TODO 4c: optional")

    # This proves Pydantic is enforcing strict data types, not just ranges.
    bad_payload = {**NORMAL, "amount": "high"}

    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422, "Should reject string value for numeric field 'amount'"
