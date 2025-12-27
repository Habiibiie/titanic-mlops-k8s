from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
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
    logger.info("Redis connection successful! 🚀")
except Exception as e:
    logger.warning(f"Could not connect to Redis, caching is disabled: {e}")
    r = None

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="An forecasting service developed using MLOps best practices.",
    version="1.0.0"
)

# --- PROMETHEUS ---
Instrumentator().instrument(app).expose(app)


# --- 1. Data Validation ---
class PassengerData(BaseModel):
    PassengerId: int = Field(..., description="Passenger ID (We want the pipeline wiper so the format isn't corrupted)")
    Name: str = Field(..., description="Passenger's Name")
    Pclass: int = Field(..., ge=1, le=3, description="Ticket Class (must be 1, 2 or 3)")
    Sex: str = Field(..., pattern="^(male|female)$", description="Gender ('male' or 'female')")
    Age: float = Field(..., ge=0, le=120, description="Age (0-120)")
    SibSp: int = Field(0, ge=0, description="Number of Siblings/Spouses")
    Parch: int = Field(0, ge=0, description="Number of Parents/Children")
    Ticket: str = Field("Unknown", description="Ticket Number")
    Fare: float = Field(..., ge=0, description="Ticket Price")
    Cabin: str = Field(None, description="Cabin Number")
    Embarked: str = Field("S", pattern="^(S|C|Q)$", description="Boarding Port (S, C, Q)")

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
        # 1. Creating Unique Key
        data_dict = passenger.dict()
        data_str = json.dumps(data_dict, sort_keys=True)
        cache_key = hashlib.sha256(data_str.encode()).hexdigest()

        # 2. Redis Check (Cache Hit)
        if r:
            cached_result = r.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT! Redis is responding: {passenger.Name}")
                return json.loads(cached_result)

        # 3. Cache Miss
        logger.info(f"Cache MISS. Model is running: {passenger.Name}")
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