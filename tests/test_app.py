from fastapi.testclient import TestClient
import sys
import os

# Proje ana dizinini yola ekle ki 'src' modülünü bulabilsin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app

client = TestClient(app)


def test_read_root():
    """Health Check testi: API ayakta mı?"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "Titanic API", "version": "1.0.0"}


def test_predict_survival():
    """Tahmin testi: Örnek bir yolcu ile model çalışıyor mu?"""
    payload = {
        "PassengerId": 123,
        "Name": "Test Passenger",
        "Pclass": 3,
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "TestTicket",
        "Fare": 7.25,
        "Cabin": "C123",
        "Embarked": "S"
    }

    response = client.post("/predict", json=payload)

    # 1. İstek başarılı mı?
    assert response.status_code == 200

    # 2. Cevap formatı doğru mu?
    json_data = response.json()
    assert "prediction" in json_data
    assert "success" in json_data
    assert json_data["success"] is True