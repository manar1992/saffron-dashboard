import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Saffron Dashboard", layout="wide", initial_sidebar_state="expanded")

PRIMARY_COLOR = "#7B3F00"
BG_CARD = "#20262e"
TEXT_COLOR = "#FFF"

def card_style():
    return f"""
        background: {BG_CARD};
        border-radius: 18px;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 3px 16px rgba(0,0,0,0.07);
        color: {TEXT_COLOR};
    """

# ----------- Helper Functions -----------
def get_growth_stage(month):
    """Return saffron growth stage name based on month."""
    if month in [8, 9, 10]:  # August, September, October
        return "Dormancy"
    elif month == 11:  # November
        return "Growth Stimulation"
    elif month in [12, 1]:  # December, January
        return "Vegetative Growth"
    elif month == 2:  # February
        return "Flowering"
    elif month in [3, 4]:  # March, April
        return "Corm Multiplication"
    elif month == 5:  # May
        return "Leaf Yellowing & Dormancy Preparation"
    else:
        return "Dormancy"

IDEAL = {
    "ph_min": 6.0, "ph_max": 8.0,
    "temperature_min": 15, "temperature_max": 25,
    "humidity_min": 40, "humidity_max": 60,
    "st_min": 15, "st_max": 25,
    "sh_min": 40, "sh_max": 60,
    "n_min": 20, "n_max": 60,
    "p_min": 60, "p_max": 80,
    "k_min": 40, "k_max": 60,
}

# ---------- Data Loading ----------
file_path = "saffron_greenhouse_synthetic_2years.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"🚨 Error loading file '{file_path}': {e}")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
df['date_only'] = df['date'].dt.date
df['hour'] = df['date'].dt.hour
df['month'] = df['date'].dt.month
df['growth_stage'] = df['month'].apply(get_growth_stage)

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("<h2 style='color:#FFA500;'>🌱 Saffron Dashboard</h2>", unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", df['date_only'].min())
    time_slider = st.slider("🕒 Select Hour:", 0, 23, step=1)

    # Growth Stage badge
    selected_row = df[(df['date_only'] == selected_date) & (df['hour'] == time_slider)]
    if not selected_row.empty:
        growth_stage = selected_row['growth_stage'].values[0]
        st.markdown(
            f"""<div style="background:#223; color:#FFD700; padding:0.32rem 0.95rem; border-radius:11px; margin-top:0.7rem; display:inline-block; font-size:1.1rem;">
            🌱 <b>Stage:</b> {growth_stage}
            </div>""",
            unsafe_allow_html=True
        )
    st.markdown("<hr style='border:1px solid #7B3F00; margin-top:1.5rem;'>", unsafe_allow_html=True)

filtered_df = df[(df['date_only'] == selected_date) & (df['hour'] == time_slider)]

# ========== Main Dashboard ==========

st.markdown(
    f"""
    <div style="padding-bottom:0.7rem;">
        <h1 style="display:inline; color:{PRIMARY_COLOR}; margin-right:12px;">🌱 Saffron Cultivation Dashboard</h1>
        <span style="font-size:1.1rem; color:#ffd97d;">| Smart monitoring for optimal saffron growth</span>
    </div>
    """,
    unsafe_allow_html=True
)

if not filtered_df.empty:
    row = filtered_df.iloc[0]
    # --- Mini Cards (Temperature, Humidity, pH) in one row ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div style='{card_style()} text-align:center'><div style='font-size:2rem;'>🌡️</div><div style='font-size:1.5rem; font-weight:bold;'>{row['temperature']:.2f} <span style='font-size:1rem; color:#bbb;'>°C</span></div><div style='font-size:1rem; color:#F7CA70;'>Temperature</div></div>",
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<div style='{card_style()} text-align:center'><div style='font-size:2rem;'>💧</div><div style='font-size:1.5rem; font-weight:bold;'>{row['humidity']:.2f} <span style='font-size:1rem; color:#bbb;'>%</span></div><div style='font-size:1rem; color:#F7CA70;'>Humidity</div></div>",
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            f"<div style='{card_style()} text-align:center'><div style='font-size:2rem;'>🧪</div><div style='font-size:1.5rem; font-weight:bold;'>{row['ph']:.2f}</div><div style='font-size:1rem; color:#F7CA70;'>pH</div></div>",
            unsafe_allow_html=True)

    # --- Crop Health Status & Story (ALWAYS on top) ---
    # ========== LOGIC ==========
    # during dormancy: always healthy, no irrigation/fertilization
    story_txt = ""
    status_color = "#4CAF50"
    crop_health_txt = "Healthy"
    if row['growth_stage'] == "Dormancy":
        story_txt = "🌿 The saffron is in dormancy stage. No irrigation or fertilization is needed. The soil parameters are optimal."
        status_color = "#4CAF50"
        crop_health_txt = "Healthy"
    else:
        # check for possible issues (non-dormancy)
        alerts = []
        if not (IDEAL["temperature_min"] <= row['temperature'] <= IDEAL["temperature_max"]):
            alerts.append("🌡️ Temperature out of range!")
        if not (IDEAL["humidity_min"] <= row['humidity'] <= IDEAL["humidity_max"]):
            alerts.append("💧 Humidity out of range!")
        if not (IDEAL["ph_min"] <= row['ph'] <= IDEAL["ph_max"]):
            alerts.append("🧪 pH out of range!")
        if row['n'] < IDEAL["n_min"]:
            alerts.append("🪴 Nitrogen is low.")
        if row['p'] < IDEAL["p_min"]:
            alerts.append("🪴 Phosphorus is low.")
        if row['k'] < IDEAL["k_min"]:
            alerts.append("🪴 Potassium is low.")

        if alerts:
            story_txt = "<br>".join(alerts)
            crop_health_txt = "Needs Attention"
            status_color = "#e53935"
        else:
            story_txt = "🌿 The saffron plant is thriving in optimal conditions. No immediate actions are required. 😊"
            crop_health_txt = "Healthy"
            status_color = "#4CAF50"

    st.markdown(
        f"""
        <div style="{card_style()} background:#254161; margin-bottom: 0.3rem;">
            <span style="font-size:1.23rem; color:{status_color}; font-weight:bold;">🌱 Crop Health: {crop_health_txt}</span><br>
            <div style="margin-top:0.45rem; font-size:1.10rem; color:#fff;">{story_txt}</div>
        </div>
        """, unsafe_allow_html=True
    )

    # --- Soil Details Table ---
    soil_params = ["n", "p", "k", "st", "sh"]
    current_values = [float(row[param]) for param in soil_params]
    recommendations, status, reasons = [], [], []
    for param, value in zip(soil_params, current_values):
        if row['growth_stage'] == "Dormancy":
            # NO irrigation in dormancy
            if param == "sh":
                recommendations.append("No irrigation (Dormancy)")
                status.append("Good")
                reasons.append("")
            elif param == "st":
                if not (IDEAL["st_min"] <= value <= IDEAL["st_max"]):
                    recommendations.append("Adjust soil temp")
                    status.append("Check")
                    reasons.append("Soil temp out of range")
                else:
                    recommendations.append("Optimal")
                    status.append("Good")
                    reasons.append("")
            else:
                recommendations.append("Optimal")
                status.append("Good")
                reasons.append("")
        else:
            if param == "n":
                if value < IDEAL["n_min"]:
                    recommendations.append("Add N to ≥20 kg/ha")
                    status.append("Bad")
                    reasons.append("Low nitrogen")
                elif value > IDEAL["n_max"]:
                    recommendations.append("Reduce N")
                    status.append("Check")
                    reasons.append("High nitrogen")
                else:
                    recommendations.append("Optimal")
                    status.append("Good")
                    reasons.append("")
            elif param == "p":
                if value < IDEAL["p_min"]:
                    recommendations.append("Add P to ≥60 kg/ha")
                    status.append("Bad")
                    reasons.append("Low phosphorus")
                elif value > IDEAL["p_max"]:
                    recommendations.append("Reduce P")
                    status.append("Check")
                    reasons.append("High phosphorus")
                else:
                    recommendations.append("Optimal")
                    status.append("Good")
                    reasons.append("")
            elif param == "k":
                if value < IDEAL["k_min"]:
                    recommendations.append("Add K to ≥40 kg/ha")
                    status.append("Bad")
                    reasons.append("Low potassium")
                elif value > IDEAL["k_max"]:
                    recommendations.append("Reduce K")
                    status.append("Check")
                    reasons.append("High potassium")
                else:
                    recommendations.append("Optimal")
                    status.append("Good")
                    reasons.append("")
            elif param == "st":
                if not (IDEAL["st_min"] <= value <= IDEAL["st_max"]):
                    recommendations.append("Adjust soil temp")
                    status.append("Check")
                    reasons.append("Soil temp out of range")
                else:
                    recommendations.append("Optimal")
                    status.append("Good")
                    reasons.append("")
            elif param == "sh":
                # التوصية بكمية الماء لو ناقصة فقط في غير مرحلة السبات
                if value < IDEAL["sh_min"]:
                    irrigation_amount = int(row['irrigation_amount']) if 'irrigation_amount' in row else 0
                    recommendations.append(f"Add water: {irrigation_amount} ml")
                    status.append("Needs Water")
                    reasons.append("Soil moisture is low")
                else:
                    recommendations.append("Optimal")
                    status.append("Good")
                    reasons.append("")

    soil_df = pd.DataFrame({
        "Parameter": soil_params,
        "Current Value": current_values,
        "Recommendation": recommendations,
        "Status": status,
        "Reason": reasons,
    })
    st.dataframe(soil_df, hide_index=True, use_container_width=True)
else:
    st.warning("⚠️ No data available for the selected time.")

# End dashboard

