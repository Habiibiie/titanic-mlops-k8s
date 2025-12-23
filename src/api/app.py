from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import sys

# Kendi modüllerimizi ekleyelim
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pipelines.prediction_pipeline import make_prediction
from src.utils.logger import get_logger

# Logger'ı başlatalım
logger = get_logger("API")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="MLOps best practices kullanılarak hazırlanmış tahmin servisi.",
    version="1.0.0"
)


# --- 1. Pydantic ile Veri Doğrulama (Data Validation) ---
# Burası MLOps'un kapı bekçisidir. Yanlış veri içeri giremez.
class PassengerData(BaseModel):
    PassengerId: int = Field(..., description="Yolcu ID (Pipeline silecek ama format bozulmasın diye istiyoruz)")
    Name: str = Field(..., description="Yolcunun Adı")
    Pclass: int = Field(..., ge=1, le=3, description="Bilet Sınıfı (1, 2 veya 3 olmalı)")
    Sex: str = Field(..., pattern="^(male|female)$", description="Cinsiyet ('male' veya 'female')")
    Age: float = Field(..., ge=0, le=120, description="Yaş (0-120 arası)")
    SibSp: int = Field(0, ge=0, description="Kardeş/Eş Sayısı")
    Parch: int = Field(0, ge=0, description="Ebeveyn/Çocuk Sayısı")
    Ticket: str = Field("Unknown", description="Bilet Numarası")
    Fare: float = Field(..., ge=0, description="Bilet Fiyatı")
    Cabin: str = Field(None, description="Kabin Numarası")
    Embarked: str = Field("S", pattern="^(S|C|Q)$", description="Biniş Limanı (S, C, Q)")

    class Config:
        # Swagger UI'da görünecek örnek veri
        json_schema_extra = {
            "example": {
                "PassengerId": 123,
                "Name": "Enes Guler",
                "Pclass": 3,
                "Sex": "male",
                "Age": 25.5,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "A/5 21171",
                "Fare": 7.25,
                "Cabin": "C123",
                "Embarked": "S"
            }
        }


# Model yolunu belirle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'titanic_pipeline.pkl')


@app.get("/")
def read_root():
    """Health Check Endpoint"""
    return {"status": "healthy", "service": "Titanic API", "version": "1.0.0"}


@app.post("/predict")
def predict_survival(passenger: PassengerData):
    """
    Tek bir yolcu için hayatta kalma tahmini yapar.
    """
    try:
        logger.info(f"API isteği alındı: {passenger.Name}")

        # Pydantic modelini dictionary'e çevir
        data_dict = passenger.dict()

        # Inference modülünü çağır
        result = make_prediction(data_dict, MODEL_PATH)

        return {
            "passenger_name": passenger.Name,
            "prediction": result,
            "success": True
        }

    except Exception as e:
        logger.error(f"API Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Localde test etmek için
    uvicorn.run(app, host="0.0.0.0", port=8000)