from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="Auto Price Predictor API")

model = CatBoostRegressor()
model.load_model('models/catboost_car_price_model.cbm')

# Завантажуємо збережені категорії
try:
    categories = joblib.load('models/valid_categories.pkl')
except FileNotFoundError:
    categories = {}


# Структура даних для одного авто
class CarFeatures(BaseModel):
    Mark: str
    Model: str
    Mileage: float
    Gearbox: str
    Age: int
    Fuel_Type: str
    Engine_Capacity: float
    Km_per_Year: float
    is_EV: int
    is_suspicious_mileage: int


# Структура даних для запиту графіка знецінення
class DepreciationRequest(BaseModel):
    car: CarFeatures
    annual_mileage: float
    years: int = 5


def process_prediction(raw_value):
    """
    Допоміжна функція, яка розпізнає, чи повернула модель логарифм ціни,
    чи вже готову ціну в доларах.
    """
    if raw_value < 50:
        price = np.exp(raw_value)
    else:
        price = raw_value

    if np.isinf(price) or np.isnan(price):
        return 0.0

    return float(price)


@app.get("/categories")
def get_categories():
    if not categories:
        raise HTTPException(status_code=404, detail="Категорії не знайдені")
    return categories


@app.post("/predict")
def predict_price(car: CarFeatures):
    input_data = pd.DataFrame([car.model_dump()])
    raw_prediction = model.predict(input_data)[0]

    predicted_price = process_prediction(raw_prediction)

    return {"predicted_price_usd": round(predicted_price, 2)}


@app.post("/predict_depreciation")
def predict_depreciation(req: DepreciationRequest):
    """
    Прогнозує ціну авто на кілька років вперед, враховуючи старіння
    та збільшення пробігу.
    """
    base_car = req.car.model_dump()
    predictions = []

    # Робимо прогноз від поточного року (0) до вказаної кількості років (5)
    for year in range(req.years + 1):
        current_car = base_car.copy()

        current_car['Age'] += year
        current_car['Mileage'] += req.annual_mileage * year

        current_car['Km_per_Year'] = current_car['Mileage'] / (current_car['Age'] + 1)

        df = pd.DataFrame([current_car])
        raw_pred = model.predict(df)[0]

        pred_price = process_prediction(raw_pred)

        if len(predictions) > 0:
            previous_price = predictions[-1]["Price"]
            if pred_price > previous_price:
                pred_price = previous_price

        predictions.append({
            "Year": year,
            "Price": round(pred_price, 2)
        })

    return {"depreciation": predictions}