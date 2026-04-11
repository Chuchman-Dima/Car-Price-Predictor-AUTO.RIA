import streamlit as st
import numpy as np
import requests
import pandas as pd
import altair as alt
import math
import time

st.set_page_config(page_title="Прогноз ціни авто", layout="centered")

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8080")

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "pred_price" not in st.session_state:
    st.session_state.pred_price = 0.0
if "payload" not in st.session_state:
    st.session_state.payload = {}

@st.cache_data(show_spinner=False)
def load_categories():
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BACKEND_URL}/categories", timeout=120)
            if response.status_code == 200:
                return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                return None
    return None


if "categories_loaded" not in st.session_state:
    with st.status("З'єднання з сервером... (це може зайняти до 2 хвилин)", expanded=True) as status:
        valid_categories = load_categories()
        if valid_categories:
            st.session_state.valid_categories = valid_categories
            st.session_state.categories_loaded = True
            status.update(label="З'єднання встановлено!", state="complete", expanded=False)
        else:
            status.update(label="Помилка підключення", state="error")
            st.error("Бекенд не відповідає. Спробуйте оновити сторінку через хвилину.")
            st.stop()
else:
    valid_categories = st.session_state.valid_categories

#  Розпаковка даних
valid_marks = valid_categories.get('valid_marks', [])
mark_model_mapping = valid_categories.get('mark_model_mapping', {})
engine_mapping = valid_categories.get('engine_mapping', {})
fuel_mapping = valid_categories.get('fuel_mapping', {})
gearbox_mapping = valid_categories.get('gearbox_mapping', {})

# Дефолтні значення
default_fuels = ["Бензин", "Дизель", "Електро", "Газ", "Гібрид (HEV)"]
default_capacities = np.arange(1.0, 8.2, 0.2).round(1).tolist()
default_gearboxes = ["Автомат", "Ручна / Механіка", "Робот", "Варіатор", "Тіптронік", "Редуктор"]

st.title("🚗 Калькулятор вартості авто")
st.write("Прогноз ціни на основі штучного інтелекту.")

col1, col2 = st.columns(2)

with col1:
    mark = st.selectbox("Марка автомобіля", sorted(valid_marks) + ['Інша'])
    available_models = ['Інша'] if mark == 'Інша' else mark_model_mapping.get(mark, []) + ['Інша']
    model_name = st.selectbox("Модель автомобіля", available_models)
    year = st.number_input("Рік випуску", min_value=1990, max_value=2026, value=2020, step=1)
    mileage = st.number_input("Пробіг (тис. км)", min_value=0, max_value=1000, value=100, step=5)

with col2:
    available_gearboxes = gearbox_mapping.get(mark, {}).get(model_name, default_gearboxes) or default_gearboxes
    gearbox = st.selectbox("Коробка передач", available_gearboxes)

    available_fuels = fuel_mapping.get(mark, {}).get(model_name, default_fuels) or default_fuels
    available_fuels = [f for f in available_fuels if f not in ['Не вказано', 'Other', '']] or ["Бензин"]
    fuel_type = st.selectbox("Тип пального", available_fuels)

    if fuel_type == 'Електро':
        st.text_input("Об'єм двигуна (л)", value="0.0 (Електро)", disabled=True)
        engine_capacity = 0.0
    else:
        available_capacities = engine_mapping.get(mark, {}).get(model_name, default_capacities) or default_capacities
        engine_capacity = st.selectbox("Об'єм двигуна (л)", available_capacities)

st.markdown("---")

if st.button("Розрахувати орієнтовну ціну", use_container_width=True):
    current_year = 2026
    age = current_year - year
    km_per_year = mileage / (age + 1)
    is_ev = 1 if fuel_type == 'Електро' else 0
    is_suspicious_mileage = 1 if (age >= 3 and km_per_year < 5) else 0

    payload = {
        "Mark": 'Other' if mark == 'Інша' else mark,
        "Model": 'Other' if model_name == 'Інша' else model_name,
        "Mileage": float(mileage),
        "Gearbox": gearbox,
        "Age": int(age),
        "Fuel_Type": fuel_type,
        "Engine_Capacity": float(engine_capacity),
        "Km_per_Year": float(km_per_year),
        "is_EV": int(is_ev),
        "is_suspicious_mileage": int(is_suspicious_mileage)
    }

    with st.spinner("Аналізуємо дані..."):
        try:
            res_price = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)

            if res_price.status_code == 200:
                data = res_price.json()
                st.session_state.pred_price = data["predicted_price_usd"]
                st.session_state.payload = payload
                st.session_state.prediction_done = True
                st.rerun()
            else:
                st.error(f"Помилка сервера: {res_price.status_code}")
        except Exception as e:
            st.error(f"Помилка з'єднання: {e}")

# Додаткові функції
if st.session_state.prediction_done:
    st.success(f"### Справедлива ринкова вартість: **${st.session_state.pred_price:,.0f}**")

    # ДИНАМІЧНЕ ПОВІДОМЛЕННЯ ПРО СКРУЧЕНИЙ ПРОБІГ
    if st.session_state.payload.get("is_suspicious_mileage") == 1:
        age_val = st.session_state.payload.get("Age")
        km_yr_val = st.session_state.payload.get("Km_per_Year")
        st.warning(
            f"⚠️ **Підозрілий пробіг:** Для авто віком {age_val} років середній пробіг виходить лише **{km_yr_val:.1f} тис. км на рік**. Зазвичай автомобілі проїжджають 10-20 тис. км щорічно. Штучний інтелект врахував можливе скручування пробігу.")

    st.markdown("---")
    st.subheader("💡 Додаткові функції аналізу")

    col_actual, col_annual = st.columns(2)
    with col_actual:
        actual_price = st.number_input("Ціна з оголошення продавця ($)", min_value=0, value=0, step=100)
    with col_annual:
        annual_mileage = st.slider("Орієнтовний пробіг за рік (тис. км)", min_value=1, max_value=100, value=15, step=1)

    if actual_price > 0:
        st.markdown("#### 🕵️ Детектор перекупів")
        diff = actual_price - st.session_state.pred_price
        diff_percent = (diff / st.session_state.pred_price) * 100

        if diff_percent > 8:
            st.error(
                f"🚨 **Ціна завищена!** Продавець просить на {diff_percent:.1f}% (${diff:,.0f}) більше за реальну вартість.")
        elif diff_percent < -8:
            st.success(f"🔥 **Вигідна пропозиція!** Ціна нижча на {abs(diff_percent):.1f}% (${abs(diff):,.0f}).")
        else:
            st.info(f"✅ **Справедлива ціна.** Запитувана вартість відповідає ринковій нормі.")

    st.markdown(f"#### 📉 Графік знецінення (при {annual_mileage} тис. км/рік)")

    depreciation_payload = {
        "car": st.session_state.payload,
        "annual_mileage": float(annual_mileage),
        "years": 5
    }

    try:
        res_depr = requests.post(f"{BACKEND_URL}/predict_depreciation", json=depreciation_payload)
        if res_depr.status_code == 200:
            depr_data = res_depr.json().get("depreciation", [])

            if depr_data:
                df_graph = pd.DataFrame(depr_data)
                df_graph["Рік"] = df_graph["Year"].apply(lambda x: "Зараз" if x == 0 else f"Через {x} р.")

                max_price = df_graph["Price"].max()
                min_price = df_graph["Price"].min()

                if max_price > 20000:
                    step = 5000
                elif max_price > 5000:
                    step = 2000
                else:
                    step = 1000

                y_max = math.ceil(max_price / step) * step
                y_min = max(0, math.floor(min_price / step) * step)

                chart = alt.Chart(df_graph).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X('Рік', sort=None, title=''),
                    y=alt.Y('Price', scale=alt.Scale(domain=[y_min, y_max]), title='Ціна ($)'),
                    tooltip=['Рік', 'Price']
                ).properties(
                    height=450
                )

                st.altair_chart(chart, use_container_width=True)

                price_now = depr_data[0]["Price"]
                price_in_5_years = depr_data[-1]["Price"]
                total_loss = price_now - price_in_5_years

                st.warning(f"💸 Орієнтовні втрати вартості за 5 років: **${total_loss:,.0f}**")
            else:
                st.error("Дані графіка відсутні у відповіді сервера.")
        else:
            st.error("Помилка під час розрахунку знецінення.")
    except requests.exceptions.ConnectionError:
        st.error("Помилка підключення до сервера.")