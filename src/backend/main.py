from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Auto Price Predictor API")

# Завантажуємо модель при старті сервера
model = CatBoostRegressor()
model.load_model('models/catboost_car_price_model.cbm')

# Завантажуємо збережені категорії
try:
    categories = joblib.load('models/valid_categories.pkl')
except FileNotFoundError:
    categories = {}


# Описуємо структуру вхідних даних
class CarFeatures(BaseModel):
    Mark: str
    Model: str
    Mileage: int
    Gearbox: str
    Age: int
    Fuel_Type: str
    Engine_Capacity: float
    Km_per_Year: float
    is_EV: int
    is_suspicious_mileage: int


@app.get("/categories")
def get_categories():
    """Віддає фронтенду словники з марками, моделями тощо"""
    if not categories:
        raise HTTPException(status_code=404, detail="Категорії не знайдені")
    return categories


@app.post("/predict")
def predict_price(car: CarFeatures):
    # Перетворюємо вхідні дані на DataFrame
    input_data = pd.DataFrame([car.model_dump()])

    # Робимо прогноз (пам'ятаємо, що модель повертає логарифм)
    predicted_log_price = model.predict(input_data)
    predicted_price = np.expm1(predicted_log_price[0])

    return {"predicted_price_usd": round(predicted_price, 2)}