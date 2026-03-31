import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- 1. НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(page_title="Прогноз ціни авто", page_icon="🚗", layout="centered")

# URL нашого бекенду (в Docker мережі сервіс називається 'backend')
BACKEND_URL = "http://backend:8000"


# --- 2. ЗАВАНТАЖЕННЯ ДАНИХ З БЕКЕНДУ ---
@st.cache_data
def load_categories():
    try:
        response = requests.get(f"{BACKEND_URL}/categories")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Не вдалося підключитися до бекенду: {e}")
        return None
    return None


valid_categories = load_categories()

if not valid_categories:
    st.warning("Очікування підключення до сервера...")
    st.stop()

# Дістаємо словники з отриманого JSON
valid_marks = valid_categories.get('valid_marks', [])
valid_models = valid_categories.get('valid_models', [])
mark_model_mapping = valid_categories.get('mark_model_mapping', {})
engine_mapping = valid_categories.get('engine_mapping', {})
fuel_mapping = valid_categories.get('fuel_mapping', {})
gearbox_mapping = valid_categories.get('gearbox_mapping', {})

# Дефолтні списки (якщо вибрано "Інша")
default_fuels = ["Бензин", "Дизель", "Електро", "Газ", "Гібрид (HEV)"]
default_capacities = np.arange(1, 8.2, 0.2).round(1).tolist()
default_gearboxes = ["Автомат", "Ручна / Механіка", "Робот", "Варіатор", "Тіптронік", "Редуктор"]

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

    # Використовуємо 2026 рік як поточний згідно з інструкцією
    year = st.number_input("Рік випуску", min_value=1990, max_value=2026, value=2023, step=1)
    mileage = st.number_input("Пробіг (тис. км)", min_value=0, max_value=1000, value=50, step=5)

with col2:
    # --- ДИНАМІЧНА КОРОБКА ПЕРЕДАЧ ---
    available_gearboxes = default_gearboxes
    if mark != 'Інша' and model_name != 'Інша':
        available_gearboxes = gearbox_mapping.get(mark, {}).get(model_name, default_gearboxes)
        if not available_gearboxes:
            available_gearboxes = default_gearboxes

    gearbox = st.selectbox("Коробка передач", available_gearboxes)

    # --- ДИНАМІЧНИЙ ТИП ПАЛЬНОГО ---
    available_fuels = default_fuels
    if mark != 'Інша' and model_name != 'Інша':
        available_fuels = fuel_mapping.get(mark, {}).get(model_name, default_fuels)
        available_fuels = [f for f in available_fuels if f not in ['Не вказано', 'Other', '']]
        if not available_fuels:
            available_fuels = ["Бензин"]

    fuel_type = st.selectbox("Тип пального", available_fuels)

    # --- ДИНАМІЧНИЙ ОБ'ЄМ ДВИГУНА ---
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

# --- 4. КНОПКА ТА ВІДПРАВКА НА БЕКЕНД ---
st.markdown("---")
if st.button("💰 Розрахувати орієнтовну ціну", use_container_width=True):
    current_year = 2026
    age = current_year - year
    km_per_year = mileage / (age + 1)
    is_ev = 1 if fuel_type == 'Електро' else 0
    is_suspicious_mileage = 1 if (age > 10 and mileage < 50) else 0

    if km_per_year < 5 and is_ev == 0 and age > 2:
        st.warning(
            "⚠️ Зверніть увагу: вказаний пробіг є аномально низьким для віку цього авто. "
            "На реальному ринку такі автомобілі часто продаються за іншою ціною через підозру на скручений пробіг.")

    final_mark = 'Other' if mark == 'Інша' else mark
    final_model = 'Other' if model_name == 'Інша' else model_name

    # Формуємо словник (JSON) для відправки на FastAPI
    payload = {
        "Mark": final_mark,
        "Model": final_model,
        "Mileage": mileage,
        "Gearbox": gearbox,
        "Age": age,
        "Fuel_Type": fuel_type,
        "Engine_Capacity": float(engine_capacity),
        "Km_per_Year": float(km_per_year),
        "is_EV": is_ev,
        "is_suspicious_mileage": is_suspicious_mileage
    }

    with st.spinner("Штучний інтелект аналізує ринок..."):
        try:
            # Звертаємось до нашого FastAPI бекенду
            response = requests.post(f"{BACKEND_URL}/predict", json=payload)

            if response.status_code == 200:
                pred_price = response.json()["predicted_price_usd"]
                st.success(f"### Орієнтовна ринкова вартість: **${pred_price:,.0f}**")
            else:
                st.error(f"Помилка бекенду: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Помилка підключення до сервера (FastAPI не відповідає).")