import os
import streamlit as st
import numpy as np
import requests
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import json
import math
from datetime import datetime

# ─────────────────────────────────────────────
# НАЛАШТУВАННЯ СТОРІНКИ
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Прогноз ціни авто",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
CURRENT_YEAR = datetime.now().year

# ─────────────────────────────────────────────
# ГЛОБАЛЬНІ СТИЛІ
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Основна палітра */
    :root {
        --primary:    #1E88E5;
        --primary-dk: #1565C0;
        --success:    #43A047;
        --warning:    #FB8C00;
        --danger:     #E53935;
        --bg-card:    rgba(255,255,255,0.03);
        --radius:     12px;
    }

    /* Заголовок */
    .hero-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .hero-sub {
        text-align: center;
        font-size: 1.05rem;
        color: #888;
        margin-top: 4px;
    }

    /* Бейдж */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green  { background:#e8f5e9; color:#2e7d32; }
    .badge-red    { background:#ffebee; color:#c62828; }
    .badge-blue   { background:#e3f2fd; color:#1565c0; }
    .badge-orange { background:#fff3e0; color:#e65100; }

    /* Картка-метрика */
    .metric-card {
        border-radius: var(--radius);
        padding: 20px 24px;
        background: var(--bg-card);
        border: 1px solid rgba(30,136,229,0.2);
        text-align: center;
    }
    .metric-card .label { font-size: 0.82rem; color: #888; margin-bottom: 6px; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; color: #1E88E5; }
    .metric-card .sub   { font-size: 0.78rem; color: #aaa; margin-top: 4px; }

    /* Score-кільце */
    .score-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 8px;
    }
    .score-label { font-size: 0.8rem; color: #888; }

    /* Лічильник кредиту */
    .loan-result {
        border-left: 4px solid #1E88E5;
        padding: 10px 16px;
        margin-top: 10px;
        border-radius: 0 var(--radius) var(--radius) 0;
    }

    /* Кнопка-головна */
    div[data-testid="stButton"] > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 700;
        letter-spacing: 0.3px;
        padding: 0.6rem 1.2rem;
        transition: all .2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(30,136,229,.35);
    }

    /* Приховати "Made with Streamlit" */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ІНІЦІАЛІЗАЦІЯ СТАНУ
# ─────────────────────────────────────────────
_defaults = {
    "prediction_done": False,
    "pred_price": 0.0,
    "payload": {},
    "compare_list": [],
    "shap_data": {},
    "saved_cars": [],          # НОВЕ: збережені авто
    "history": [],             # НОВЕ: історія розрахунків
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ
# ─────────────────────────────────────────────
def fmt_money(amount: float, currency: str) -> str:
    """Форматує число як грошову суму."""
    return f"{int(amount):,} {currency}".replace(",", "\u202f")


def condition_score(age: int, mileage: float, fuel_type: str, gearbox: str) -> tuple[int, str, str]:
    """
    Розраховує умовний бал технічного стану (0–100).
    Повертає (бал, колір, текст-рейтинг).
    """
    score = 100

    # Вік
    if age <= 2:    score -= 0
    elif age <= 5:  score -= 10
    elif age <= 10: score -= 20
    elif age <= 15: score -= 32
    else:           score -= 48

    # Пробіг
    if mileage <= 50:    score -= 0
    elif mileage <= 100: score -= 8
    elif mileage <= 200: score -= 18
    elif mileage <= 300: score -= 28
    else:                score -= 40

    # Бонус за тип пального
    if fuel_type in ("Електро", "Гібрид (HEV)"): score += 4
    if gearbox == "Автомат":                      score += 2

    score = max(0, min(100, score))

    if score >= 80: return score, "#43A047", "Відмінний"
    if score >= 60: return score, "#FB8C00", "Хороший"
    if score >= 40: return score, "#FF7043", "Задовільний"
    return score, "#E53935", "Поганий"


def loan_monthly(principal: float, rate_pct: float, months: int) -> float:
    """Формула ануїтетного платежу."""
    if rate_pct == 0:
        return principal / months
    r = rate_pct / 100 / 12
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


# ─────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ ДАНИХ З БЕКЕНДУ
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_categories():
    for attempt in range(3):
        try:
            r = requests.get(f"{BACKEND_URL}/categories", timeout=120)
            if r.status_code == 200:
                return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < 2:
                time.sleep(5)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_exchange_rates() -> dict:
    default = {"USD": 1.0, "UAH": 43.89, "EUR": 0.852}
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if r.status_code == 200:
            d = r.json()
            return {
                "USD": 1.0,
                "UAH": round(d["rates"].get("UAH", default["UAH"]), 2),
                "EUR": round(d["rates"].get("EUR", default["EUR"]), 4),
            }
    except Exception:
        pass
    return default


# ─────────────────────────────────────────────
# LAYOUT — ГОЛОВНА КОЛОНКА
# ─────────────────────────────────────────────
spacer_left, col_main, spacer_right = st.columns([1, 8, 1])

with col_main:

    # ── Підключення до бекенду ──────────────────
    if "categories_loaded" not in st.session_state:
        with st.status("🔄 З'єднання з сервером… (до 2 хвилин)", expanded=True) as status:
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

    # ── Розпаковка категорій ─────────────────────
    valid_marks      = [m for m in valid_categories.get("valid_marks", []) if m != "Причеп"]
    mark_model_map   = valid_categories.get("mark_model_mapping", {})
    engine_mapping   = valid_categories.get("engine_mapping", {})
    fuel_mapping     = valid_categories.get("fuel_mapping", {})
    gearbox_mapping  = valid_categories.get("gearbox_mapping", {})

    default_fuels      = ["Бензин", "Дизель", "Електро", "Газ", "Гібрид (HEV)"]
    default_capacities = np.arange(1.0, 8.2, 0.2).round(1).tolist()
    default_gearboxes  = ["Автомат", "Ручна / Механіка", "Робот", "Варіатор", "Тіптронік", "Редуктор"]

    # ── ЗАГОЛОВОК ────────────────────────────────
    st.markdown("<h1 class='hero-title'>🚗 Калькулятор вартості авто</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>Штучний інтелект для визначення справедливої ринкової ціни</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ── ШВИДКЕ ЗАВАНТАЖЕННЯ ЗБЕРЕЖЕНОГО АВТО (нове) ─
    if st.session_state.saved_cars:
        with st.expander("📂 Завантажити збережене авто"):
            saved_labels = [f"{c['mark']} {c['model']} ({c['year']})" for c in st.session_state.saved_cars]
            chosen = st.selectbox("Оберіть авто:", saved_labels, key="saved_selector")
            if st.button("Завантажити параметри", key="load_saved"):
                idx = saved_labels.index(chosen)
                car = st.session_state.saved_cars[idx]
                # Записуємо у session_state щоб потрапити у віджети
                st.session_state["_preload"] = car
                st.rerun()

    preload = st.session_state.pop("_preload", None)

    # ── ФОРМА ВВОДУ ──────────────────────────────
    with st.container(border=True):
        st.subheader("📋 Характеристики автомобіля")

        col1, col2 = st.columns(2)

        with col1:
            mark_default = preload["mark"] if preload and preload["mark"] in sorted(valid_marks) else None
            mark = st.selectbox(
                "Марка автомобіля",
                sorted(valid_marks) + ["Інша"],
                index=(sorted(valid_marks) + ["Інша"]).index(mark_default) if mark_default else 0,
            )
            available_models = ["Інша"] if mark == "Інша" else mark_model_map.get(mark, []) + ["Інша"]
            model_default = preload["model"] if preload and preload["model"] in available_models else available_models[0]
            model_name = st.selectbox(
                "Модель автомобіля",
                available_models,
                index=available_models.index(model_default),
            )
            year = st.number_input(
                "Рік випуску",
                min_value=1990, max_value=CURRENT_YEAR, step=1,
                value=preload["year"] if preload else 2020,
            )
            mileage = st.number_input(
                "Пробіг (тис. км)",
                min_value=0, max_value=1000, step=5,
                value=preload["mileage"] if preload else 100,
            )

        with col2:
            available_gearboxes = gearbox_mapping.get(mark, {}).get(model_name, default_gearboxes) or default_gearboxes
            gearbox = st.selectbox("Коробка передач", available_gearboxes)

            available_fuels = fuel_mapping.get(mark, {}).get(model_name, default_fuels) or default_fuels
            available_fuels = [f for f in available_fuels if f not in ("Не вказано", "Other", "")] or ["Бензин"]
            fuel_type = st.selectbox("Тип пального", available_fuels)

            if fuel_type == "Електро":
                st.text_input("Об'єм двигуна (л)", value="0.0 (Електро)", disabled=True)
                engine_capacity = 0.0
            else:
                available_capacities = engine_mapping.get(mark, {}).get(model_name, default_capacities) or default_capacities
                engine_capacity = st.selectbox("Об'єм двигуна (л)", available_capacities)

        # ── Індикатор стану (нове) ──────────────
        age_preview = CURRENT_YEAR - year
        score, score_color, score_label = condition_score(age_preview, mileage, fuel_type, gearbox)
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin-top:8px;padding:10px 16px;
                        border-radius:8px;background:rgba(30,136,229,.06);border:1px solid rgba(30,136,229,.15)">
                <div style="font-size:2rem;font-weight:800;color:{score_color}">{score}</div>
                <div>
                    <div style="font-size:.78rem;color:#888;">Умовний бал технічного стану</div>
                    <div style="font-weight:600;color:{score_color};">{score_label}</div>
                </div>
                <div style="flex:1;height:8px;border-radius:4px;background:#eee;margin-left:8px;overflow:hidden">
                    <div style="width:{score}%;height:100%;border-radius:4px;
                                background:linear-gradient(90deg,{score_color}aa,{score_color})"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ── ЗБЕРЕЖЕННЯ АВТО (нове) ──────────────────
    save_col, btn_col, _ = st.columns([1, 2, 1])
    with save_col:
        if st.button("💾 Зберегти авто", use_container_width=True, type="secondary"):
            st.session_state.saved_cars.append({
                "mark": mark, "model": model_name,
                "year": year, "mileage": mileage,
                "fuel": fuel_type, "gearbox": gearbox,
                "engine": engine_capacity,
            })
            st.toast(f"{mark} {model_name} збережено!", icon="💾")

    with btn_col:
        calculate_btn = st.button("🚀 Розрахувати орієнтовну ціну", use_container_width=True, type="primary")

    # ── РОЗРАХУНОК ───────────────────────────────
    if calculate_btn:
        age = CURRENT_YEAR - year
        km_per_year = mileage / (age + 1)
        is_ev = 1 if fuel_type == "Електро" else 0
        is_suspicious = 1 if (age >= 3 and km_per_year < 5) else 0

        payload = {
            "Mark":                 "Other" if mark == "Інша" else mark,
            "Model":                "Other" if model_name == "Інша" else model_name,
            "Mileage":              float(mileage),
            "Gearbox":              gearbox,
            "Age":                  int(age),
            "Fuel_Type":            fuel_type,
            "Engine_Capacity":      float(engine_capacity),
            "Km_per_Year":          float(km_per_year),
            "is_EV":                int(is_ev),
            "is_suspicious_mileage": int(is_suspicious),
        }

        progress = st.progress(0, text="Аналізуємо ринкові дані…")
        try:
            for pct in (20, 50):
                time.sleep(0.15)
                progress.progress(pct)

            res_price = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
            progress.progress(90)

            if res_price.status_code == 200:
                data = res_price.json()
                st.session_state.pred_price = data["predicted_price_usd"]
                st.session_state.shap_data  = data.get("shap_values", {
                    "Рік випуску":      1500 if age < 5 else -1000,
                    "Пробіг":           -800 if mileage > 150 else 500,
                    "Тип пального":     300 if fuel_type in ("Дизель", "Гібрид (HEV)") else -100,
                    "Коробка передач":  400 if gearbox == "Автомат" else -300,
                })
                st.session_state.payload         = payload
                st.session_state.prediction_done = True

                # Зберігаємо в історію (нове)
                st.session_state.history.append({
                    "Час":            datetime.now().strftime("%H:%M:%S"),
                    "Авто":           f"{mark} {model_name}",
                    "Рік":            year,
                    "Пробіг (тис.)":  mileage,
                    "Оцінка (USD)":   int(data["predicted_price_usd"]),
                })

                progress.progress(100, text="Готово!")
                time.sleep(0.3)
                progress.empty()
                st.rerun()
            else:
                progress.empty()
                st.error(f"Помилка сервера: {res_price.status_code}")
        except Exception as e:
            progress.empty()
            st.error(f"Помилка з'єднання: {e}")

    # ════════════════════════════════════════════
    # РЕЗУЛЬТАТИ
    # ════════════════════════════════════════════
    if st.session_state.prediction_done:
        st.markdown("---")
        st.markdown("## 📊 Результати оцінки")

        rates = get_exchange_rates()

        # ── Валюта + Ціна ────────────────────────
        with st.container(border=True):
            col_curr, col_price, col_range, col_score = st.columns([1, 2, 2, 1])

            with col_curr:
                curr = st.radio(
                    "Оберіть валюту:",
                    ["USD", "UAH", "EUR"],
                    horizontal=False,
                    key="currency_radio_selector",
                )
                if curr != "USD":
                    st.caption(f"Курс: 1 USD = {rates[curr]:.2f} {curr}")

            price_usd      = st.session_state.pred_price
            price_conv     = price_usd * rates[curr]
            margin         = price_conv * 0.05

            with col_price:
                st.metric(
                    label="Справедлива ринкова вартість",
                    value=fmt_money(price_conv, curr),
                )
            with col_range:
                st.metric(
                    label="Діапазон ринкових цін",
                    value=f"Від {fmt_money(price_conv - margin, curr)}",
                    delta=f"До {fmt_money(price_conv + margin, curr)}",
                    delta_color="off",
                )

            # Бал стану у результатах
            p = st.session_state.payload
            sc, sc_color, sc_label = condition_score(
                p.get("Age", 0), p.get("Mileage", 0),
                p.get("Fuel_Type", ""), p.get("Gearbox", ""),
            )
            with col_score:
                st.markdown(
                    f"""<div class='score-wrap'>
                        <svg width='70' height='70' viewBox='0 0 70 70'>
                          <circle cx='35' cy='35' r='28' fill='none' stroke='#eee' stroke-width='8'/>
                          <circle cx='35' cy='35' r='28' fill='none' stroke='{sc_color}' stroke-width='8'
                            stroke-dasharray='{2*3.14159*28*sc/100:.1f} {2*3.14159*28:.1f}'
                            stroke-linecap='round'
                            transform='rotate(-90 35 35)'/>
                          <text x='35' y='40' text-anchor='middle'
                                font-size='16' font-weight='700' fill='{sc_color}'>{sc}</text>
                        </svg>
                        <span class='score-label'>{sc_label}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Попередження ─────────────────────────
        if p.get("is_suspicious_mileage") == 1:
            st.warning(
                f"⚠️ **Підозрілий пробіг:** Для авто віком {p.get('Age')} років середній пробіг "
                f"виходить лише **{p.get('Km_per_Year'):.1f} тис. км/рік**. ШІ врахував можливе скручування.",
                icon="⚠️",
            )

        # ── Пояснення ціни (SHAP) ────────────────
        with st.expander("🔍 Як ШІ розрахував цю ціну? (Вплив характеристик)"):
            st.write("На графіку показано, як кожна характеристика збільшує або зменшує базову вартість:")
            df_shap = pd.DataFrame(
                list(st.session_state.shap_data.items()), columns=["Характеристика", "Вплив ($)"]
            )
            df_shap["Колір"] = np.where(df_shap["Вплив ($)"] > 0, "#43A047", "#E53935")

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            ax.barh(df_shap["Характеристика"], df_shap["Вплив ($)"], color=df_shap["Колір"], height=0.6)
            ax.axvline(0, color="grey", linewidth=1.5, linestyle="--")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(colors="gray")
            ax.set_xlabel("Зміна ціни (USD)", color="gray")
            st.pyplot(fig)

        st.write("")
        st.subheader("💡 Інструменти покупця")

        # ── Детектор перекупів + Витрати ─────────
        col_tools1, col_tools2 = st.columns(2)

        with col_tools1:
            with st.container(border=True):
                st.markdown("#### 🕵️ Детектор перекупів")
                actual_price = st.number_input(
                    "Ціна з оголошення продавця (USD)",
                    min_value=0, value=0, step=100,
                )
                if actual_price > 0:
                    diff = actual_price - price_usd
                    diff_pct = diff / price_usd * 100
                    if diff_pct > 8:
                        st.error(f"🚨 **Завищена!** На {diff_pct:.1f}% (${diff:,.0f}) дорожче реальної вартості.")
                    elif diff_pct < -8:
                        st.success(f"🔥 **Вигідно!** Нижче на {abs(diff_pct):.1f}% (${abs(diff):,.0f}).")
                    else:
                        st.info("✅ **Справедлива ціна.** Відповідає ринковій нормі.")
                else:
                    st.caption("Введіть ціну, щоб перевірити адекватність.")

        with col_tools2:
            with st.container(border=True):
                st.markdown("#### ⛽ Вартість володіння")
                annual_mileage = st.slider(
                    "Ваш орієнтовний пробіг за рік (тис. км)",
                    min_value=1, max_value=100, value=15, step=1,
                )
                if p.get("is_EV") == 0:
                    eng = p.get("Engine_Capacity", 0)
                    consumption = eng * 2.5 + 2 if eng > 0 else 8
                    fuel_price_uah = 54
                    yearly_uah = (annual_mileage * 1000 / 100) * consumption * fuel_price_uah
                    yearly_usd = int(yearly_uah / rates["UAH"])
                    st.info(
                        f"Витрати на пальне: **~${yearly_usd:,}/рік**\n\n"
                        f"*(При витраті {consumption:.1f} л / 100 км)*".replace(",", " ")
                    )
                else:
                    st.success("🔋 **Електромобіль:** Витрати на зарядку значно нижчі від бензину.")

        # ══════════════════════════════════════════
        # НОВЕ: Калькулятор кредиту / лізингу
        # ══════════════════════════════════════════
        with st.container(border=True):
            st.markdown("#### 🏦 Кредитний калькулятор")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                down_pct = st.slider("Перший внесок (%)", 0, 80, 20, 5)
            with lc2:
                loan_months = st.selectbox("Термін кредиту", [12, 24, 36, 48, 60, 84], index=2)
            with lc3:
                rate_pct = st.number_input("Річна ставка (%)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)

            down_usd      = price_usd * down_pct / 100
            principal_usd = price_usd - down_usd
            monthly_usd   = loan_monthly(principal_usd, rate_pct, loan_months)
            total_pay_usd = monthly_usd * loan_months + down_usd
            overpay_usd   = total_pay_usd - price_usd

            lm1, lm2, lm3, lm4 = st.columns(4)
            lm1.metric("Перший внесок",  fmt_money(down_usd * rates[curr], curr))
            lm2.metric("Щомісячний платіж", fmt_money(monthly_usd * rates[curr], curr))
            lm3.metric("Загальна сума виплат", fmt_money(total_pay_usd * rates[curr], curr))
            lm4.metric("Переплата",      fmt_money(overpay_usd * rates[curr], curr), delta_color="inverse",
                       delta=f"-{overpay_usd/price_usd*100:.1f}%")

        # ── Графік знецінення ─────────────────────
        with st.container(border=True):
            st.markdown(f"#### 📉 Прогноз знецінення (при {annual_mileage} тис. км/рік)")
            depr_payload = {
                "car": p,
                "annual_mileage": float(annual_mileage),
                "years": 5,
            }
            try:
                res_depr = requests.post(f"{BACKEND_URL}/predict_depreciation", json=depr_payload)
                if res_depr.status_code == 200:
                    depr_data = res_depr.json().get("depreciation", [])
                    if depr_data:
                        df_graph = pd.DataFrame(depr_data)
                        df_graph["Рік"] = df_graph["Year"].apply(
                            lambda x: "Зараз" if x == 0 else f"Через {x} р."
                        )
                        chart = (
                            alt.Chart(df_graph)
                            .mark_area(
                                line={"color": "#1E88E5"},
                                color=alt.Gradient(
                                    gradient="linear",
                                    stops=[
                                        alt.GradientStop(color="#1E88E5", offset=0),
                                        alt.GradientStop(color="rgba(255,255,255,0)", offset=1),
                                    ],
                                    x1=1, x2=1, y1=1, y2=0,
                                ),
                            )
                            .encode(
                                x=alt.X("Рік", sort=None, title=""),
                                y=alt.Y("Price", scale=alt.Scale(zero=False), title="Орієнтовна ціна ($)"),
                                tooltip=["Рік", "Price"],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(chart, use_container_width=True)
                        total_loss = depr_data[0]["Price"] - depr_data[-1]["Price"]
                        st.warning(f"💸 Орієнтовні втрати вартості за 5 років: **${total_loss:,.0f}**")
            except Exception:
                pass

        # ── Порівняння авто ───────────────────────
        st.write("")
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            if st.button("➕ Додати авто до порівняння", use_container_width=True):
                car_info = {
                    "Марка/Модель":     f"{p['Mark']} {p['Model']}",
                    "Рік":              CURRENT_YEAR - p["Age"],
                    "Оцінка ШІ (USD)":  int(price_usd),
                    "Оголошення (USD)": actual_price if actual_price > 0 else "—",
                    "Бал стану":        sc,
                }
                st.session_state.compare_list.append(car_info)
                st.toast("Авто збережено для порівняння!", icon="✅")

        if st.session_state.compare_list:
            st.markdown("### 📋 Таблиця порівняння авто")
            st.dataframe(pd.DataFrame(st.session_state.compare_list), use_container_width=True)
            if st.button("🗑 Очистити порівняння", type="secondary"):
                st.session_state.compare_list = []
                st.rerun()

    # ════════════════════════════════════════════
    # НОВЕ: Історія розрахунків
    # ════════════════════════════════════════════
    if st.session_state.history:
        st.markdown("---")
        with st.expander("🕐 Історія розрахунків у цій сесії"):
            df_hist = pd.DataFrame(st.session_state.history)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

            # Кнопка очищення
            if st.button("🗑 Очистити історію", key="clear_history", type="secondary"):
                st.session_state.history = []
                st.rerun()

    # ── Підвал ────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;font-size:.78rem;color:#aaa;'>"
        "Оцінка є орієнтовною та базується на ринкових даних. "
        "Не є офіційним висновком про вартість транспортного засобу."
        "</p>",
        unsafe_allow_html=True,
    )