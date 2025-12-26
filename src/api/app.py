from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import sys
import redis
import json
import hashlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pipelines.prediction_pipeline import make_prediction
from src.utils.logger import get_logger

logger = get_logger("API")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
try:
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    r.ping()
    logger.info("Redis bağlantısı başarılı! 🚀")
except Exception as e:
    logger.warning(f"Redis'e bağlanılamadı, caching devre dışı: {e}")
    r = None

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="MLOps best practices kullanılarak hazırlanmış tahmin servisi.",
    version="1.0.0"
)


# --- 1. Data Validation ---
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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'titanic_pipeline.pkl')


@app.get("/")
def read_root():
    """Health Check Endpoint"""
    return {"status": "healthy", "service": "Titanic API", "version": "1.0.0"}


@app.post("/predict")
def predict_survival(passenger: PassengerData):
    try:
        # 1. Unique Key Oluşturma (Gelen veriyi hash'le)
        data_dict = passenger.dict()
        data_str = json.dumps(data_dict, sort_keys=True)
        cache_key = hashlib.sha256(data_str.encode()).hexdigest()

        # 2. Redis Kontrolü (Cache Hit)
        if r:
            cached_result = r.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT! Redis'ten cevap dönülüyor: {passenger.Name}")
                return json.loads(cached_result)

        # 3. Cache Miss (Tahmin Yap)
        logger.info(f"Cache MISS. Model çalıştırılıyor: {passenger.Name}")
        result = make_prediction(data_dict, MODEL_PATH)

        response_payload = {
            "passenger_name": passenger.Name,
            "prediction": result,
            "success": True,
            "source": "model"
        }

        # 4. Sonucu Redis'e Yaz (1 Saatlik ömür verelim - 3600 sn)
        if r:
            cache_payload = response_payload.copy()
            cache_payload["source"] = "cache"
            r.setex(cache_key, 3600, json.dumps(cache_payload))

        return response_payload

    except Exception as e:
        logger.error(f"API Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)