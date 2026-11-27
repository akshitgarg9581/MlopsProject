from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _valid_payload() -> dict:
    return {
        "flu_cases_reported": 120,
        "temp_c": 24.5,
        "nearby_clinics": 5,
        "discount_pct": 10.0,
        "stock_lines": 1200,
        "distance_to_hospital_km": 3.2,
        "shop_type": "chain",
        "locality": "Old Town",
    }


def test_health_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid() -> None:
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)


def test_predict_rejects_high_flu_cases() -> None:
    payload = _valid_payload()
    payload["flu_cases_reported"] = 351
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_nearby_clinics() -> None:
    payload = _valid_payload()
    payload["nearby_clinics"] = -1
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
