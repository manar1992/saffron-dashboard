import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Saffron Dashboard", layout="wide", initial_sidebar_state="expanded")

PRIMARY_COLOR = "#7B3F00"
SECONDARY_COLOR = "#FFD700"
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

def stat_card(icon, label, value, unit):
    st.markdown(
        f"""
        <div style="{card_style()} text-align:center; display:flex; flex-direction:column; align-items:center; min-width:110px;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-size:1.5rem; font-weight:bold;">{value} <span style="font-size:1rem; color:#bbb;">{unit}</span></div>
            <div style="font-size:1rem; margin-top:0.08rem; color:#F7CA70;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============= الدالة لتحويل الشهر إلى Growth Stage ============
def get_growth_stage(month):
    if month in [8, 9, 10]:
        return "Dormancy"
    elif month == 11:
        return "Growth Stimulation"
    elif month in [12, 1]:
        return "Vegetative Growth"
    elif month == 2:
        return "Flowering"
    elif month in [3, 4]:
        return "Corm Multiplication"
    elif month == 5:
        return "Leaf Yellowing & Dormancy Preparation"
    else:
        return "Unknown"

# ============= تحميل البيانات ============
file_path = "saffron_greenhouse_synthetic_2years.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"🚨 Error loading file '{file_path}': {e}")
    st.stop()

df['datetime'] = pd.to_datetime(df['date'])
df['date_only'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month

IDEAL = {
    "ph_min": 6.0, "ph_max": 8.0,
    "temperature_min": 15, "temperature_max": 25,
    "humidity_min": 40, "humidity_max": 60,
    "n_min": 20, "n_max": 60,
    "p_min": 60, "p_max": 80,
    "k_min": 40, "k_max": 60,
}

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("<h2 style='color:#FFA500;'>🌱 Saffron Dashboard</h2>", unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", df['date_only'].min())
    time_slider = st.slider("🕒 Select Hour:", 0, 23, step=1)

    selected_row = df[(df['date_only'] == selected_date) & (df['hour'] == time_slider)]
    if not selected_row.empty:
        row_month = int(selected_row['month'].values[0])
        growth_stage = get_growth_stage(row_month)
        st.markdown(
            f"""<div style="background:#223; color:#FFD700; padding:0.32rem 0.95rem; border-radius:11px; margin-top:0.7rem; display:inline-block; font-size:1.06rem;">
            🌱 <b>Stage:</b> {growth_stage}
            </div>""",
            unsafe_allow_html=True
        )
    else:
        growth_stage = "Unknown"
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
    col1, col2, col3 = st.columns(3)
    with col1:
        stat_card("🌡️", "Temperature", f"{filtered_df['temperature'].values[0]:.2f}", "°C")
    with col2:
        stat_card("💧", "Humidity", f"{filtered_df['humidity'].values[0]:.2f}", "%")
    with col3:
        stat_card("🧪", "pH", f"{filtered_df['ph'].values[0]:.2f}", "")
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No data available for the selected time.")

# -------------- القواعد الزراعية للتحكم بالتوصيات -----------------
def get_irrigation_recommendation(growth_stage, sh, irrigation_amount):
    """تعطي توصية الري حسب المرحلة"""
    if growth_stage == "Dormancy":
        return "No irrigation needed", "No water", "Good"
    elif growth_stage == "Leaf Yellowing & Dormancy Preparation":
        return "Irrigate every two weeks (1-2 L/m²)", f"Add water: {int(irrigation_amount)} ml" if irrigation_amount > 0 else "No water", "Conditional"
    elif growth_stage == "Growth Stimulation":
        return "Irrigate every 10-14 days (2-3 L/m²)", f"Add water: {int(irrigation_amount)} ml" if irrigation_amount > 0 else "No water", "Conditional"
    elif growth_stage == "Vegetative Growth":
        return "Twice per week (3-5 L/m²)", f"Add water: {int(irrigation_amount)} ml" if irrigation_amount > 0 else "No water", "Conditional"
    elif growth_stage == "Flowering":
        return "Once per week (2-3 L/m²)", f"Add water: {int(irrigation_amount)} ml" if irrigation_amount > 0 else "No water", "Conditional"
    elif growth_stage == "Corm Multiplication":
        return "Irrigate every 10 days (3-4 L/m²)", f"Add water: {int(irrigation_amount)} ml" if irrigation_amount > 0 else "No water", "Conditional"
    else:
        return "Follow expert advice", "-", "-"

if not filtered_df.empty:
    # --------- مثال: الحصول على stage من الشهر ----------
    row_month = int(filtered_df['month'].values[0])
    growth_stage = get_growth_stage(row_month)

    # --------- قراءة القيم الحالية ---------
    sh = filtered_df['sh'].values[0]
    irrigation_amount = filtered_df['irrigation_amount'].values[0]

    # --------- منطق توصية الري ----------
    irrigation_freq, irrigation_detail, irrigation_status = get_irrigation_recommendation(growth_stage, sh, irrigation_amount)

    # ------------ Soil Table بدون تكرار pH -------------
    soil_params = ["n", "p", "k", "st", "sh"] # حذف ph
    current_values = [float(filtered_df[param].values[0]) for param in soil_params]
    recommendations, status, reasons = [], [], []
    for param, value in zip(soil_params, current_values):
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
            if not (18 <= value <= 22):
                recommendations.append("Adjust soil temp")
                status.append("Check")
                reasons.append("Soil temp out of range")
            else:
                recommendations.append("Optimal")
                status.append("Good")
                reasons.append("")
        elif param == "sh":
            if growth_stage == "Dormancy":
                recommendations.append("No irrigation (Dormancy)")
                status.append("Good")
                reasons.append("")
            elif value < IDEAL["humidity_min"]:
                recommendations.append(irrigation_detail)
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
    st.markdown(f"<div style='{card_style()}background:#1A212B;'><span style='font-size:1.19rem;'>🪴 <b>Soil Details</b></span></div>", unsafe_allow_html=True)
    st.dataframe(soil_df, hide_index=True, use_container_width=True)

    # ------------ Alerts & Recommendations ------------
    alerts = []
    if growth_stage == "Dormancy":
        alerts.append("🌱 Plant is in Dormancy stage. No irrigation or fertilization needed.")
    else:
        if not (IDEAL["humidity_min"] <= filtered_df['humidity'].values[0] <= IDEAL["humidity_max"]):
            alerts.append("💧 <b>Humidity out of range!</b> Adjust irrigation (40–60%).")
        if not (IDEAL["temperature_min"] <= filtered_df['temperature'].values[0] <= IDEAL["temperature_max"]):
            alerts.append("🌡️ <b>Temperature out of range!</b> Optimal: 15–25°C.")
        if filtered_df['n'].values[0] < IDEAL["n_min"]:
            alerts.append("🪴 <b>Nitrogen is low.</b> (Add N to reach at least 20 kg/ha).")
        if filtered_df['p'].values[0] < IDEAL["p_min"]:
            alerts.append("🪴 <b>Phosphorus is low.</b> (Add P to reach at least 60 kg/ha).")
        if filtered_df['k'].values[0] < IDEAL["k_min"]:
            alerts.append("🪴 <b>Potassium is low.</b> (Add K to reach at least 40 kg/ha).")
    if alerts:
        st.markdown(
            f"""
            <div style="{card_style()}background:#391E1A;">
                <span style="font-size:1.2rem; color:#FFD700;"><b>⚠️ Alerts & Recommendations</b></span>
                <ul style="margin-top:0.7rem;">
                {''.join([f"<li style='margin-bottom:0.4rem;'>{a}</li>" for a in alerts])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------- Plant Story Card ----------
    if growth_stage == "Dormancy":
        story_txt = "🌱 Saffron is dormant. No actions required."
    else:
        story_txt = "🌿 The saffron plant is thriving in optimal conditions. Monitor alerts for optimal growth."
    st.markdown(
        f"""
        <div style="{card_style()}background:#254161;">
            <div style="margin-top:0.7rem; font-size:1.1rem;">
                {story_txt}
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # ----------- Smooth Temperature Chart -----------
    st.markdown("<div style='margin-top:1.3rem;'>", unsafe_allow_html=True)
    temp_data = df[df['date_only'] == selected_date]['temperature'].values
    hour_data = df[df['date_only'] == selected_date]['hour'].values
    smooth_temp = gaussian_filter1d(temp_data, sigma=2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hour_data, y=smooth_temp, mode='lines', name='Temperature (Smoothed)', line=dict(width=4, color="#FFD700")))
    fig.add_trace(go.Scatter(x=hour_data, y=temp_data, mode='markers', name='Original', marker=dict(size=6, color="#7B3F00"), opacity=0.4))
    fig.update_layout(
        title="🌡️ Temperature Trend",
        xaxis_title="Hour",
        yaxis_title="Temperature (°C)",
        plot_bgcolor=BG_CARD,
        paper_bgcolor="#181C21",
        font=dict(color=TEXT_COLOR),
        margin=dict(l=30, r=30, t=50, b=30),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
