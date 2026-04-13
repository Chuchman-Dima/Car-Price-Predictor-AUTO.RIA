import os
import streamlit as st
import numpy as np
import requests
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import time

# --- НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(page_title="Прогноз ціни авто", page_icon="🚗", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# --- ІНІЦІАЛІЗАЦІЯ СТАНУ ---
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "pred_price" not in st.session_state:
    st.session_state.pred_price = 0.0
if "payload" not in st.session_state:
    st.session_state.payload = {}
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []
if "shap_data" not in st.session_state:
    st.session_state.shap_data = {}


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


# --- РОЗМІТКА СТОРІНКИ (80% ШИРИНИ) ---
# Створюємо 3 колонки: пуста (10%), головна (80%), пуста (10%)
spacer_left, col_main, spacer_right = st.columns([1, 8, 1])

with col_main:
    if "categories_loaded" not in st.session_state:
        with st.status("🔄 З'єднання з сервером... (це може зайняти до 2 хвилин)", expanded=True) as status:
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

    # --- РОЗПАКОВКА ДАНИХ ---
    valid_marks = valid_categories.get('valid_marks', [])
    valid_marks = [mark for mark in valid_marks if mark != 'Причеп']
    mark_model_mapping = valid_categories.get('mark_model_mapping', {})
    engine_mapping = valid_categories.get('engine_mapping', {})
    fuel_mapping = valid_categories.get('fuel_mapping', {})
    gearbox_mapping = valid_categories.get('gearbox_mapping', {})

    default_fuels = ["Бензин", "Дизель", "Електро", "Газ", "Гібрид (HEV)"]
    default_capacities = np.arange(1.0, 8.2, 0.2).round(1).tolist()
    default_gearboxes = ["Автомат", "Ручна / Механіка", "Робот", "Варіатор", "Тіптронік", "Редуктор"]

    # --- ЗАГОЛОВОК ---
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🚗 Калькулятор вартості авто</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>Штучний інтелект для визначення справедливої ринкової ціни</p>",
        unsafe_allow_html=True)
    st.write("")

    # --- ІНТЕРФЕЙС ВВОДУ ---
    with st.container(border=True):
        st.subheader("📋 Характеристики автомобіля")

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
                available_capacities = engine_mapping.get(mark, {}).get(model_name,
                                                                        default_capacities) or default_capacities
                engine_capacity = st.selectbox("Об'єм двигуна (л)", available_capacities)

    st.write("")

    # Кнопка розрахунку по центру
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        calculate_btn = st.button("🚀 Розрахувати орієнтовну ціну", use_container_width=True, type="primary")

    if calculate_btn:
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

        with st.spinner("Аналізуємо ринкові дані..."):
            try:
                res_price = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
                if res_price.status_code == 200:
                    data = res_price.json()
                    st.session_state.pred_price = data["predicted_price_usd"]

                    st.session_state.shap_data = data.get("shap_values", {
                        "Рік випуску": 1500 if age < 5 else -1000,
                        "Пробіг": -800 if mileage > 150 else 500,
                        "Тип пального": 300 if fuel_type in ["Дизель", "Гібрид (HEV)"] else -100,
                        "Коробка передач": 400 if gearbox == "Автомат" else -300
                    })

                    st.session_state.payload = payload
                    st.session_state.prediction_done = True
                    st.rerun()
                else:
                    st.error(f"Помилка сервера: {res_price.status_code}")
            except Exception as e:
                st.error(f"Помилка з'єднання: {e}")

    # --- ВІДОБРАЖЕННЯ РЕЗУЛЬТАТІВ ---
    if st.session_state.prediction_done:
        st.markdown("---")
        st.markdown("## 📊 Результати оцінки")

        # 1. ГОЛОВНА МЕТРИКА ЦІНИ
        rates = {"USD": 1.0, "UAH": 39.5, "EUR": 0.92}

        with st.container(border=True):
            col_curr, col_price, col_range = st.columns([1, 2, 2])

            with col_curr:
                curr = st.radio("Оберіть валюту:", ["USD", "UAH", "EUR"], horizontal=False)

            price_converted = st.session_state.pred_price * rates[curr]
            margin = price_converted * 0.05

            with col_price:
                st.metric(
                    label="Справедлива ринкова вартість",
                    value=f"{int(price_converted):,} {curr}".replace(",", " ")
                )
            with col_range:
                st.metric(
                    label="Діапазон ринкових цін",
                    value=f"Від {int(price_converted - margin):,} {curr}".replace(",", " "),
                    delta=f"До {int(price_converted + margin):,} {curr}".replace(",", " "),
                    delta_color="off"
                )

        if st.session_state.payload.get("is_suspicious_mileage") == 1:
            st.warning(
                f"⚠️ **Підозрілий пробіг:** Для авто віком {st.session_state.payload.get('Age')} років середній пробіг виходить лише **{st.session_state.payload.get('Km_per_Year'):.1f} тис. км на рік**. ШІ врахував можливе скручування.",
                icon="⚠️"
            )

        # 2. ПОЯСНЕННЯ ЦІНИ (SHAP)
        with st.expander("🔍 Як ШІ розрахував цю ціну? (Вплив характеристик)"):
            st.write("На графіку показано, як кожна характеристика збільшує або зменшує базову вартість:")
            df_shap = pd.DataFrame(list(st.session_state.shap_data.items()), columns=['Характеристика', 'Вплив ($)'])
            df_shap['Колір'] = np.where(df_shap['Вплив ($)'] > 0, 'green', 'red')

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_alpha(0.0)  # Прозорий фон графіка
            ax.patch.set_alpha(0.0)

            ax.barh(df_shap['Характеристика'], df_shap['Вплив ($)'], color=df_shap['Колір'], height=0.6)
            ax.axvline(0, color='grey', linewidth=1.5, linestyle='--')

            # Стилізація осей
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')
            ax.xaxis.label.set_color('gray')

            ax.set_xlabel('Зміна ціни (USD)')
            st.pyplot(fig)

        st.write("")
        st.subheader("💡 Інструменти покупця")

        # 3. АНАЛІЗАТОР ТА ВИТРАТИ
        col_tools1, col_tools2 = st.columns(2)

        with col_tools1:
            with st.container(border=True):
                st.markdown("#### 🕵️ Детектор перекупів")
                actual_price = st.number_input("Введіть ціну з оголошення продавця (USD)", min_value=0, value=0,
                                               step=100)

                if actual_price > 0:
                    diff = actual_price - st.session_state.pred_price
                    diff_percent = (diff / st.session_state.pred_price) * 100

                    if diff_percent > 8:
                        st.error(
                            f"🚨 **Ціна завищена!** Продавець просить на {diff_percent:.1f}% (${diff:,.0f}) більше за реальну вартість.")
                    elif diff_percent < -8:
                        st.success(
                            f"🔥 **Вигідна пропозиція!** Ціна нижча на {abs(diff_percent):.1f}% (${abs(diff):,.0f}).")
                    else:
                        st.info(f"✅ **Справедлива ціна.** Запитувана вартість відповідає ринковій нормі.")
                else:
                    st.caption("Введіть ціну, щоб перевірити наскільки вона адекватна.")

        with col_tools2:
            with st.container(border=True):
                st.markdown("#### ⛽ Вартість володіння")
                annual_mileage = st.slider("Ваш орієнтовний пробіг за рік (тис. км)", min_value=1, max_value=100,
                                           value=15, step=1)

                if st.session_state.payload.get("is_EV") == 0:
                    eng_cap = st.session_state.payload.get("Engine_Capacity")
                    est_consumption = eng_cap * 2.5 + 2 if eng_cap > 0 else 8
                    fuel_price_uah = 54
                    yearly_cost_uah = (annual_mileage * 1000 / 100) * est_consumption * fuel_price_uah

                    st.info(
                        f"Орієнтовні витрати на пальне: **~${int(yearly_cost_uah / rates['UAH']):,}** на рік.\n\n*(При середній витраті {est_consumption:.1f} л / 100 км)*".replace(
                            ",", " "))
                else:
                    st.success(
                        "🔋 **Електромобіль:** Витрати на зарядку значно нижчі і залежать від тарифу (вдома чи на швидкісних станціях).")

        # 4. ГРАФІК ЗНЕЦІНЕННЯ
        with st.container(border=True):
            st.markdown(f"#### 📉 Прогноз знецінення авто (при {annual_mileage} тис. км/рік)")
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

                        chart = alt.Chart(df_graph).mark_area(
                            line={'color': '#1E88E5'},
                            color=alt.Gradient(
                                gradient='linear',
                                stops=[alt.GradientStop(color='#1E88E5', offset=0),
                                       alt.GradientStop(color='rgba(255,255,255,0)', offset=1)],
                                x1=1, x2=1, y1=1, y2=0
                            )
                        ).encode(
                            x=alt.X('Рік', sort=None, title=''),
                            y=alt.Y('Price', scale=alt.Scale(zero=False), title='Орієнтовна ціна ($)'),
                            tooltip=['Рік', 'Price']
                        ).properties(height=350)

                        st.altair_chart(chart, use_container_width=True)

                        total_loss = depr_data[0]["Price"] - depr_data[-1]["Price"]
                        st.warning(f"💸 Орієнтовні втрати вартості за 5 років: **${total_loss:,.0f}**")
                else:
                    st.error("Помилка під час розрахунку знецінення.")
            except Exception:
                pass

                # 5. ПОРІВНЯННЯ АВТО
        st.write("")
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            if st.button("➕ Додати авто до порівняння", use_container_width=True):
                car_info = {
                    "Марка/Модель": f"{st.session_state.payload['Mark']} {st.session_state.payload['Model']}",
                    "Рік": st.session_state.payload['Age'] * -1 + 2026,
                    "Оцінка ШІ (USD)": int(st.session_state.pred_price),
                    "Оголошення (USD)": actual_price if actual_price > 0 else "-"
                }
                st.session_state.compare_list.append(car_info)
                st.toast("Авто збережено для порівняння!", icon="✅")

        if st.session_state.compare_list:
            st.markdown("### 📋 Таблиця порівняння авто")
            st.dataframe(pd.DataFrame(st.session_state.compare_list), use_container_width=True)

            if st.button("🗑 Очистити порівняння", type="secondary"):
                st.session_state.compare_list = []
                st.rerun()