import os
import time
from datetime import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

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
    :root {
        --primary:    #1E88E5;
        --primary-dk: #1565C0;
        --success:    #43A047;
        --warning:    #FB8C00;
        --danger:     #E53935;
        --bg-card:    rgba(255,255,255,0.03);
        --radius:     12px;
    }
    .hero-title {
        text-align: center; font-size: 2.4rem; font-weight: 800;
        letter-spacing: -0.5px; color: #1E88E5; margin-bottom: 0;
    }
    .hero-sub {
        text-align: center; font-size: 1.05rem; color: #888; margin-top: 4px;
    }
    .score-wrap {
        display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px;
    }
    .score-label { font-size: 0.8rem; color: #888; }
    div[data-testid="stButton"] > button[kind="primary"] {
        border-radius: 8px; font-weight: 700; letter-spacing: 0.3px;
        padding: 0.6rem 1.2rem; transition: all .2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(30,136,229,.35);
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ІНІЦІАЛІЗАЦІЯ СТАНУ
# ─────────────────────────────────────────────
_defaults = {
    "prediction_done": False,
    "pred_price":      0.0,
    "payload":         {},
    "compare_list":    [],
    "shap_data":       {},
    "saved_cars":      [],
    "history":         [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ
# ─────────────────────────────────────────────
def fmt_money(amount: float, currency: str) -> str:
    return f"{int(amount):,} {currency}".replace(",", "\u202f")


def condition_score(age: int, mileage: float, fuel_type: str, gearbox: str) -> tuple[int, str, str]:
    score = 100
    if age <= 2:       score -= 0
    elif age <= 5:     score -= 10
    elif age <= 10:    score -= 20
    elif age <= 15:    score -= 32
    else:              score -= 48

    if mileage <= 50:    score -= 0
    elif mileage <= 100: score -= 8
    elif mileage <= 200: score -= 18
    elif mileage <= 300: score -= 28
    else:                score -= 40

    if fuel_type in ("Електро", "Гібрид (HEV)"): score += 4
    if gearbox == "Автомат":                      score += 2
    score = max(0, min(100, score))

    if score >= 80: return score, "#43A047", "Відмінний"
    if score >= 60: return score, "#FB8C00", "Хороший"
    if score >= 40: return score, "#FF7043", "Задовільний"
    return score, "#E53935", "Поганий"


def loan_monthly(principal: float, rate_pct: float, months: int) -> float:
    if rate_pct == 0 or months == 0:
        return principal / max(months, 1)
    r = rate_pct / 100 / 12
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def score_ring_svg(sc: int, sc_color: str, sc_label: str) -> str:
    circ = 2 * 3.14159 * 28
    dash = circ * sc / 100
    return f"""<div class='score-wrap'>
        <svg width='70' height='70' viewBox='0 0 70 70'>
          <circle cx='35' cy='35' r='28' fill='none' stroke='#eee' stroke-width='8'/>
          <circle cx='35' cy='35' r='28' fill='none' stroke='{sc_color}' stroke-width='8'
            stroke-dasharray='{dash:.1f} {circ:.1f}' stroke-linecap='round'
            transform='rotate(-90 35 35)'/>
          <text x='35' y='40' text-anchor='middle'
                font-size='16' font-weight='700' fill='{sc_color}'>{sc}</text>
        </svg>
        <span class='score-label'>{sc_label}</span>
    </div>"""


# ─────────────────────────────────────────────
# ЗАПИТИ ДО БЕКЕНДУ
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_categories() -> dict | None:
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
# ГОЛОВНИЙ LAYOUT
# ─────────────────────────────────────────────
spacer_left, col_main, spacer_right = st.columns([1, 8, 1])

with col_main:

    # ── Підключення до бекенду ──────────────────
    # ВИПРАВЛЕННЯ: завантаження відбувається тільки у фронтенді,
    # st.status захищений від None через окремий placeholder.
    if "categories_loaded" not in st.session_state:
        _status_placeholder = st.empty()
        with _status_placeholder.status(
            "🔄 З'єднання з сервером… (до 2 хвилин)", expanded=True
        ) as _status:
            _cats = load_categories()
            if _cats:
                st.session_state.valid_categories = _cats
                st.session_state.categories_loaded = True
                _status.update(label="✅ З'єднання встановлено!", state="complete", expanded=False)
            else:
                _status.update(label="❌ Помилка підключення", state="error")
                st.error(
                    f"Бекенд не відповідає. Спробуйте оновити сторінку.\n\n"
                    f"URL бекенду: `{BACKEND_URL}`"
                )
                st.stop()
        valid_categories = st.session_state.valid_categories
    else:
        valid_categories = st.session_state.valid_categories

    # ── Розпаковка категорій ─────────────────────
    valid_marks     = [m for m in valid_categories.get("valid_marks", []) if m != "Причеп"]
    mark_model_map  = valid_categories.get("mark_model_mapping", {})
    engine_mapping  = valid_categories.get("engine_mapping", {})
    fuel_mapping    = valid_categories.get("fuel_mapping", {})
    gearbox_mapping = valid_categories.get("gearbox_mapping", {})

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

    # ── Завантаження збереженого авто ───────────
    if st.session_state.saved_cars:
        with st.expander("📂 Завантажити збережене авто"):
            saved_labels = [f"{c['mark']} {c['model']} ({c['year']})" for c in st.session_state.saved_cars]
            col_sel, col_del = st.columns([3, 1])
            with col_sel:
                chosen = st.selectbox("Оберіть авто:", saved_labels, key="saved_selector")
            with col_del:
                st.write("")
                if st.button("🗑 Видалити", key="del_saved", type="secondary"):
                    idx = saved_labels.index(chosen)
                    st.session_state.saved_cars.pop(idx)
                    st.rerun()
            if st.button("⬇️ Завантажити параметри", key="load_saved"):
                idx = saved_labels.index(chosen)
                st.session_state["_preload"] = st.session_state.saved_cars[idx]
                st.rerun()

    preload = st.session_state.pop("_preload", None)

    # ── ФОРМА ВВОДУ ──────────────────────────────
    with st.container(border=True):
        st.subheader("📋 Характеристики автомобіля")
        col1, col2 = st.columns(2)

        with col1:
            mark_list = sorted(valid_marks) + ["Інша"]
            mark_default = preload["mark"] if preload and preload["mark"] in mark_list else mark_list[0]
            mark = st.selectbox("Марка автомобіля", mark_list,
                                index=mark_list.index(mark_default))

            available_models = (["Інша"] if mark == "Інша"
                                else mark_model_map.get(mark, []) + ["Інша"])
            model_default = (preload["model"] if preload and preload["model"] in available_models
                             else available_models[0])
            model_name = st.selectbox("Модель автомобіля", available_models,
                                      index=available_models.index(model_default))

            year = st.number_input("Рік випуску",
                                   min_value=1990, max_value=CURRENT_YEAR, step=1,
                                   value=int(preload["year"]) if preload else 2020)
            mileage = st.number_input("Пробіг (тис. км)",
                                      min_value=0, max_value=1000, step=5,
                                      value=int(preload["mileage"]) if preload else 100)

        with col2:
            available_gearboxes = (gearbox_mapping.get(mark, {}).get(model_name, default_gearboxes)
                                   or default_gearboxes)
            gearbox = st.selectbox("Коробка передач", available_gearboxes)

            available_fuels = (fuel_mapping.get(mark, {}).get(model_name, default_fuels)
                               or default_fuels)
            available_fuels = ([f for f in available_fuels
                                if f not in ("Не вказано", "Other", "")] or ["Бензин"])
            fuel_type = st.selectbox("Тип пального", available_fuels)

            if fuel_type == "Електро":
                st.text_input("Об'єм двигуна (л)", value="0.0 (Електро)", disabled=True)
                engine_capacity = 0.0
            else:
                available_caps = (engine_mapping.get(mark, {}).get(model_name, default_capacities)
                                  or default_capacities)
                engine_capacity = st.selectbox("Об'єм двигуна (л)", available_caps)

        # ── Індикатор стану ──────────────────────
        age_preview = CURRENT_YEAR - year
        sc0, sc0_color, sc0_label = condition_score(age_preview, mileage, fuel_type, gearbox)
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;margin-top:8px;
                            padding:10px 16px;border-radius:8px;
                            background:rgba(30,136,229,.06);border:1px solid rgba(30,136,229,.15)">
                <div style="font-size:2rem;font-weight:800;color:{sc0_color}">{sc0}</div>
                <div>
                    <div style="font-size:.78rem;color:#888;">Умовний бал технічного стану</div>
                    <div style="font-weight:600;color:{sc0_color};">{sc0_label}</div>
                </div>
                <div style="flex:1;height:8px;border-radius:4px;background:#eee;
                            margin-left:8px;overflow:hidden">
                    <div style="width:{sc0}%;height:100%;border-radius:4px;
                                background:linear-gradient(90deg,{sc0_color}aa,{sc0_color})"></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # ── КНОПКИ ───────────────────────────────────
    save_col, btn_col, _ = st.columns([1, 2, 1])
    with save_col:
        if st.button("💾 Зберегти авто", use_container_width=True, type="secondary"):
            st.session_state.saved_cars.append({
                "mark": mark, "model": model_name, "year": year,
                "mileage": mileage, "fuel": fuel_type,
                "gearbox": gearbox, "engine": engine_capacity,
            })
            st.toast(f"{mark} {model_name} збережено!", icon="💾")
    with btn_col:
        calculate_btn = st.button("🚀 Розрахувати орієнтовну ціну",
                                  use_container_width=True, type="primary")

    # ── РОЗРАХУНОК ───────────────────────────────
    if calculate_btn:
        age          = CURRENT_YEAR - year
        km_per_year  = mileage / (age + 1)
        is_ev        = 1 if fuel_type == "Електро" else 0
        is_suspicious = 1 if (age >= 3 and km_per_year < 5) else 0

        payload = {
            "Mark":                  "Other" if mark == "Інша" else mark,
            "Model":                 "Other" if model_name == "Інша" else model_name,
            "Mileage":               float(mileage),
            "Gearbox":               gearbox,
            "Age":                   int(age),
            "Fuel_Type":             fuel_type,
            "Engine_Capacity":       float(engine_capacity),
            "Km_per_Year":           float(km_per_year),
            "is_EV":                 int(is_ev),
            "is_suspicious_mileage": int(is_suspicious),
        }

        progress = st.progress(0, text="Аналізуємо ринкові дані…")
        try:
            for pct in (20, 50):
                time.sleep(0.12)
                progress.progress(pct)

            res = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
            progress.progress(90)

            if res.status_code == 200:
                data = res.json()
                st.session_state.pred_price      = data["predicted_price_usd"]
                st.session_state.shap_data       = data.get("shap_values", {})
                st.session_state.payload         = payload
                st.session_state.prediction_done = True

                st.session_state.history.append({
                    "Час":           datetime.now().strftime("%H:%M:%S"),
                    "Авто":          f"{mark} {model_name}",
                    "Рік":           year,
                    "Пробіг (тис.)": mileage,
                    "Оцінка (USD)":  int(data["predicted_price_usd"]),
                })

                progress.progress(100, text="Готово!")
                time.sleep(0.3)
                progress.empty()
                st.rerun()

            elif res.status_code == 422:
                progress.empty()
                st.error(f"⚠️ Помилка валідації: {res.json().get('detail', 'Некоректні параметри.')}")
            elif res.status_code == 503:
                progress.empty()
                st.error("🔧 Модель ще не готова. Зачекайте кілька секунд і спробуйте ще раз.")
            else:
                progress.empty()
                st.error(f"Помилка сервера: {res.status_code} — {res.text[:300]}")

        except requests.exceptions.Timeout:
            progress.empty()
            st.error("⏱ Сервер не відповів за 60 секунд. Спробуйте ще раз.")
        except requests.exceptions.ConnectionError:
            progress.empty()
            st.error(f"🔌 Не вдалося підключитися до бекенду (`{BACKEND_URL}`).")
        except Exception as e:
            progress.empty()
            st.error(f"Несподівана помилка: {e}")

    # ════════════════════════════════════════════
    # РЕЗУЛЬТАТИ
    # ════════════════════════════════════════════
    if st.session_state.prediction_done:
        st.markdown("---")
        st.markdown("## 📊 Результати оцінки")

        rates = get_exchange_rates()
        p     = st.session_state.payload

        # ── Валюта + Ціна ────────────────────────
        with st.container(border=True):
            col_curr, col_price, col_range, col_score = st.columns([1, 2, 2, 1])

            with col_curr:
                curr = st.radio("Валюта:", ["USD", "UAH", "EUR"],
                                horizontal=False, key="currency_radio_selector")
                if curr != "USD":
                    st.caption(f"1 USD = {rates[curr]:.2f} {curr}")

            price_usd  = st.session_state.pred_price
            price_conv = price_usd * rates[curr]
            margin     = price_conv * 0.05

            with col_price:
                st.metric("Справедлива ринкова вартість", fmt_money(price_conv, curr))
            with col_range:
                st.metric(
                    "Діапазон ринкових цін",
                    value=f"Від {fmt_money(price_conv - margin, curr)}",
                    delta=f"До {fmt_money(price_conv + margin, curr)}",
                    delta_color="off",
                )

            sc, sc_color, sc_label = condition_score(
                p.get("Age", 0), p.get("Mileage", 0),
                p.get("Fuel_Type", ""), p.get("Gearbox", ""),
            )
            with col_score:
                st.markdown(score_ring_svg(sc, sc_color, sc_label), unsafe_allow_html=True)

        # ── Попередження про пробіг ──────────────
        if p.get("is_suspicious_mileage") == 1:
            st.warning(
                f"⚠️ **Підозрілий пробіг:** Для авто віком {p.get('Age')} р. середній пробіг "
                f"лише **{p.get('Km_per_Year'):.1f} тис. км/рік**. ШІ врахував можливе скручування.",
                icon="⚠️",
            )

        # ── SHAP ─────────────────────────────────
        if st.session_state.shap_data:
            with st.expander("🔍 Як ШІ розрахував цю ціну?"):
                st.caption("Вплив кожної характеристики на прогнозовану ціну:")
                df_shap = pd.DataFrame(
                    list(st.session_state.shap_data.items()),
                    columns=["Характеристика", "Вплив ($)"],
                ).sort_values("Вплив ($)")
                df_shap["Колір"] = np.where(df_shap["Вплив ($)"] > 0, "#43A047", "#E53935")

                fig, ax = plt.subplots(figsize=(10, max(3, len(df_shap) * 0.6)))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                ax.barh(df_shap["Характеристика"], df_shap["Вплив ($)"],
                        color=df_shap["Колір"], height=0.55)
                ax.axvline(0, color="grey", linewidth=1.2, linestyle="--")
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                ax.tick_params(colors="gray")
                ax.set_xlabel("Зміна ціни (USD)", color="gray")
                st.pyplot(fig)

        st.write("")
        st.subheader("💡 Інструменти покупця")

        col_tools1, col_tools2 = st.columns(2)

        with col_tools1:
            with st.container(border=True):
                st.markdown("#### 🕵️ Детектор перекупів")
                actual_price = st.number_input(
                    "Ціна з оголошення (USD)", min_value=0, value=0, step=100,
                )
                if actual_price > 0:
                    diff     = actual_price - price_usd
                    diff_pct = diff / price_usd * 100
                    if diff_pct > 8:
                        st.error(f"🚨 **Завищена!** На {diff_pct:.1f}% (${diff:,.0f}) дорожче.")
                        st.caption(
                            f"💬 Рекомендована ціна для торгу: "
                            f"**${int(price_usd * 0.97):,}**".replace(",", " ")
                        )
                    elif diff_pct < -8:
                        st.success(f"🔥 **Вигідно!** Нижче на {abs(diff_pct):.1f}% (${abs(diff):,.0f}).")
                    else:
                        st.info("✅ **Справедлива ціна.** Відповідає ринку.")
                else:
                    st.caption("Введіть ціну продавця, щоб перевірити.")

        with col_tools2:
            with st.container(border=True):
                st.markdown("#### ⛽ Вартість пального")
                annual_mileage = st.slider(
                    "Пробіг за рік (тис. км)", min_value=1, max_value=100, value=15, step=1,
                )
                if p.get("is_EV") == 0:
                    eng         = p.get("Engine_Capacity", 0)
                    consumption = eng * 2.5 + 2 if eng > 0 else 8
                    yearly_uah  = (annual_mileage * 1000 / 100) * consumption * 54
                    yearly_usd  = int(yearly_uah / rates["UAH"])
                    st.info(
                        f"Витрати на пальне: **~${yearly_usd:,}/рік**\n\n"
                        f"*(Витрата ~{consumption:.1f} л / 100 км)*".replace(",", " ")
                    )
                else:
                    ev_uah = (annual_mileage * 1000 / 100) * 18 * 4.32
                    ev_usd = int(ev_uah / rates["UAH"])
                    st.success(
                        f"🔋 **Електромобіль.** Витрати на зарядку: **~${ev_usd:,}/рік**\n\n"
                        f"*(~18 кВт·год / 100 км)*".replace(",", " ")
                    )

        # ── Кредитний калькулятор ─────────────────
        with st.container(border=True):
            st.markdown("#### 🏦 Кредитний калькулятор")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                down_pct = st.slider("Перший внесок (%)", 0, 80, 20, 5)
            with lc2:
                loan_months = st.selectbox("Термін", [12, 24, 36, 48, 60, 84], index=2)
            with lc3:
                rate_pct = st.number_input("Ставка (%/рік)", 0.0, 50.0, 15.0, 0.5)

            down_usd      = price_usd * down_pct / 100
            principal_usd = price_usd - down_usd
            monthly_usd   = loan_monthly(principal_usd, rate_pct, loan_months)
            total_pay_usd = monthly_usd * loan_months + down_usd
            overpay_usd   = total_pay_usd - price_usd

            lm1, lm2, lm3, lm4 = st.columns(4)
            lm1.metric("Перший внесок",     fmt_money(down_usd    * rates[curr], curr))
            lm2.metric("Щомісячний платіж", fmt_money(monthly_usd * rates[curr], curr))
            lm3.metric("Загальна сума",     fmt_money(total_pay_usd * rates[curr], curr))
            lm4.metric("Переплата",         fmt_money(overpay_usd * rates[curr], curr),
                       delta=f"+{overpay_usd / price_usd * 100:.1f}%", delta_color="inverse")

        # ── Графік знецінення ─────────────────────
        with st.container(border=True):
            st.markdown(f"#### 📉 Прогноз знецінення (при {annual_mileage} тис. км/рік)")
            try:
                res_depr = requests.post(
                    f"{BACKEND_URL}/predict_depreciation",
                    json={"car": p, "annual_mileage": float(annual_mileage), "years": 5},
                    timeout=30,
                )
                if res_depr.status_code == 200:
                    body      = res_depr.json()
                    depr_data = body.get("depreciation", [])
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
                                y=alt.Y("Price", scale=alt.Scale(zero=False), title="Ціна ($)"),
                                tooltip=["Рік", "Price"],
                            )
                            .properties(height=280)
                        )
                        st.altair_chart(chart, use_container_width=True)

                        total_loss = body.get(
                            "total_loss_usd",
                            depr_data[0]["Price"] - depr_data[-1]["Price"],
                        )
                        st.warning(
                            f"💸 Втрата вартості за 5 років: "
                            f"**{fmt_money(total_loss * rates[curr], curr)}**"
                        )
            except Exception:
                st.caption("Графік знецінення тимчасово недоступний.")

        # ── Порівняння авто ───────────────────────
        st.write("")
        col_add, _ = st.columns([1, 2])
        with col_add:
            if st.button("➕ Додати до порівняння", use_container_width=True):
                st.session_state.compare_list.append({
                    "Марка/Модель":     f"{p['Mark']} {p['Model']}",
                    "Рік":              CURRENT_YEAR - p["Age"],
                    "Оцінка ШІ (USD)":  int(price_usd),
                    "Оголошення (USD)": actual_price if actual_price > 0 else "—",
                    "Бал стану":        sc,
                })
                st.toast("Авто додано до порівняння!", icon="✅")

        if st.session_state.compare_list:
            st.markdown("### 📋 Таблиця порівняння")
            st.dataframe(pd.DataFrame(st.session_state.compare_list), use_container_width=True)
            if st.button("🗑 Очистити порівняння", type="secondary"):
                st.session_state.compare_list = []
                st.rerun()

    # ── Історія розрахунків ───────────────────────
    if st.session_state.history:
        st.markdown("---")
        with st.expander("🕐 Історія розрахунків у цій сесії"):
            st.dataframe(
                pd.DataFrame(st.session_state.history),
                use_container_width=True, hide_index=True,
            )
            if st.button("🗑 Очистити історію", key="clear_history", type="secondary"):
                st.session_state.history = []
                st.rerun()

    # ── Підвал ────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;font-size:.78rem;color:#aaa;'>"
        "Оцінка є орієнтовною та базується на ринкових даних. "
        "Не є офіційним висновком про вартість транспортного засобу."
        "</p>",
        unsafe_allow_html=True,
    )