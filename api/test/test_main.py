from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from api.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}
def test_valid_prediction_request():
    request_data = {
        "date": (datetime.now().strftime('%Y-%m-%d')),
        "exogenus": [1, 2, 3],
        "steps": 6
    }
    response = client.post("/v1/prediction", json=request_data)
    assert response.status_code == 200
    assert "predcit" in response.json()

def test_invalid_month_prediction_request():
    request_data = {
        "date": (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
        "exogenus": [1, 2, 3],
        "steps": 6
    }
    response = client.post("/v1/prediction", json=request_data)
    assert response.status_code == 400
    assert "Invalid date" in response.json()["detail"]

def test_invalid_steps_prediction_request():
    request_data = {
        "date": (datetime.now().strftime('%Y-%m-%d')),
        "exogenus": [1, 2, 3],
        "steps": 20
    }
    response = client.post("/v1/prediction", json=request_data)
    assert response.status_code == 400
    assert "Step must be greater than 0 and less than 12" in response.json()["detail"]