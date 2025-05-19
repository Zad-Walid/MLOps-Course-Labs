
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_health_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    data ={
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["Churn", "Not Churn"]



