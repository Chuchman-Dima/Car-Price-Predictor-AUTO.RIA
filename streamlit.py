import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import datetime

# --- 1. НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(page_title="Прогноз ціни авто", page_icon="🚗", layout="centered")


# --- 2. ЗАВАНТАЖЕННЯ МОДЕЛІ ТА ДАНИХ ---
@st.cache_resource
def load_assets():
    model = CatBoostRegressor()
    model.load_model('models/catboost_car_price_model.cbm')
    valid_categories = joblib.load('models/valid_categories.pkl')
    return model, valid_categories


model, valid_categories = load_assets()
valid_marks = valid_categories['valid_marks']
valid_models = valid_categories['valid_models']

# Дістаємо наші словники
mark_model_mapping = valid_categories.get('mark_model_mapping', {})
engine_mapping = valid_categories.get('engine_mapping', {})
fuel_mapping = valid_categories.get('fuel_mapping', {})  # <-- Дістаємо типи пального

# Стандартні списки на випадок, якщо вибрано "Інша"
default_fuels = ["Бензин", "Дизель", "Електро", "Газ", "Газ / Бензин", "Гібрид (HEV)"]
default_capacities = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 3.0]

# --- 3. ІНТЕРФЕЙС КОРИСТУВАЧА ---
st.title("🚗 Калькулятор вартості вживаного авто")
st.write("Введіть параметри автомобіля, і штучний інтелект спрогнозує його ціну на основі реальних даних з ринку.")

col1, col2 = st.columns(2)

with col1:
    mark = st.selectbox("Марка автомобіля", sorted(valid_marks) + ['Інша'])

    if mark == 'Інша':
        available_models = ['Інша']
    else:
        available_models = mark_model_mapping.get(mark, []) + ['Інша']

    model_name = st.selectbox("Модель автомобіля", available_models)

    year = st.number_input("Рік випуску", min_value=2000, max_value=datetime.datetime.now().year, value=2018, step=1)
    mileage = st.number_input("Пробіг (тис. км)", min_value=0, max_value=1000, value=150, step=10)

with col2:
    gearbox = st.selectbox("Коробка передач", ["Автомат", "Ручна / Механіка"])

    # ДИНАМІЧНИЙ ТИП ПАЛЬНОГО
    # ДИНАМІЧНИЙ ТИП ПАЛЬНОГО
    if mark != 'Інша' and model_name != 'Інша':
        available_fuels = fuel_mapping.get(mark, {}).get(model_name, default_fuels)

        # Додаткова перевірка: прибираємо все, що містить цифри або "л."
        available_fuels = [f for f in available_fuels if not any(char.isdigit() for char in f) and 'л.' not in f]
        available_fuels = [f for f in available_fuels if f not in ['Не вказано', 'Unknown', '']]

        if not available_fuels:
            available_fuels = ["Бензин"]  # Дефолт, якщо після очистки порожньо
    else:
        available_fuels = default_fuels

    # Якщо залишився лише 1 варіант (наприклад, тільки "Електро" для Tesla), блокуємо вибір
    is_strictly_one = len(available_fuels) == 1

    fuel_type = st.selectbox("Тип пального", available_fuels, disabled=is_strictly_one)

    # ДИНАМІЧНИЙ ОБ'ЄМ ДВИГУНА
    if fuel_type == 'Електро':
        st.text_input("Об'єм двигуна (л)", value="0.0 (Електро)", disabled=True)
        engine_capacity = 0.0
    else:
        if mark != 'Інша' and model_name != 'Інша':
            available_capacities = engine_mapping.get(mark, {}).get(model_name, default_capacities)
            if not available_capacities:
                available_capacities = default_capacities
        else:
            available_capacities = default_capacities

        engine_capacity = st.selectbox("Об'єм двигуна (л)", available_capacities)

# --- 4. КНОПКА ТА ЛОГІКА ПРОГНОЗУ ---
st.markdown("---")
if st.button("💰 Розрахувати орієнтовну ціну", use_container_width=True):

    current_year = datetime.datetime.now().year
    age = current_year - year
    km_per_year = mileage / (age + 1)
    is_ev = 1 if fuel_type == 'Електро' else 0
    is_suspicious_mileage = 1 if (age > 10 and mileage < 50) else 0

    if km_per_year < 5 and is_ev == 0:
        st.warning(
            "⚠️ Зверніть увагу: вказаний пробіг є аномально низьким для віку цього авто. На реальному ринку такі автомобілі часто продаються дешевше через підозру на скручений пробіг або історію ДТП.")

    final_mark = mark if mark in valid_marks else 'Other'
    final_model = model_name if model_name in valid_models else 'Other'

    features = ['Mark', 'Model', 'Mileage', 'Gearbox', 'Age', 'Fuel_Type', 'Engine_Capacity', 'Km_per_Year', 'is_EV',
                'is_suspicious_mileage']

    input_data = pd.DataFrame([[
        final_mark, final_model, mileage, gearbox, age, fuel_type, engine_capacity, km_per_year, is_ev,
        is_suspicious_mileage
    ]], columns=features)

    pred_log = model.predict(input_data)
    pred_price = np.expm1(pred_log)[0]

    st.success(f"### Орієнтовна ринкова вартість: **${pred_price:,.0f}**")