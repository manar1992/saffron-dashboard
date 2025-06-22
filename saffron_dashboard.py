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

# ========== Data Load ==========
file_path = "saffron_greenhouse_synthetic_2years.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"🚨 Error loading file '{file_path}': {e}")
    st.stop()

df['datetime'] = pd.to_datetime(df['date'])
df['date_only'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

IDEAL = {
    "ph_min": 6.0, "ph_max": 8.0,
    "temperature_min": 15, "temperature_max": 25,
    "humidity_min": 40, "humidity_max": 60,
    "n_min": 20, "n_max": 60,
    "p_min": 60, "p_max": 80,
    "k_min": 40, "k_max": 60,
    "st_min": 18, "st_max": 22,
    "sh_min": 40, "sh_max": 60,
}

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("<h2 style='color:#FFA500;'>🌱 Saffron Dashboard</h2>", unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", df['date_only'].min())
    time_slider = st.slider("🕒 Select Hour:", 0, 23, step=1)
    selected_row = df[(df['date_only'] == selected_date) & (df['hour'] == time_slider)]
    if not selected_row.empty:
        growth_stage = selected_row['stage'].values[0]
        st.markdown(
            f"""<div style="background:#223; color:#FFD700; padding:0.32rem 0.95rem; border-radius:11px; margin-top:0.7rem; display:inline-block; font-size:1.06rem;">
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

# --- Mini Cards ---
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

# ========== Logic for Health & Story ==========

def crop_health_logic(row):
    if row['stage'] == "Dormancy":
        return "Healthy"
    # if any major value out of ideal, mark attention
    elif not (IDEAL["ph_min"] <= row['ph'] <= IDEAL["ph_max"]):
        return "At Risk"
    elif not (IDEAL["temperature_min"] <= row['temperature'] <= IDEAL["temperature_max"]):
        return "Needs Attention"
    elif not (IDEAL["humidity_min"] <= row['humidity'] <= IDEAL["humidity_max"]):
        return "Needs Attention"
    elif row['n'] < IDEAL["n_min"]:
        return "Needs Attention"
    elif row['p'] < IDEAL["p_min"]:
        return "Needs Attention"
    elif row['k'] < IDEAL["k_min"]:
        return "Needs Attention"
    else:
        return "Healthy"

if not filtered_df.empty:
    current_row = filtered_df.iloc[0]
    predicted_health = crop_health_logic(current_row)
    health_color = "#4CAF50" if predicted_health == "Healthy" else "#ff9800" if predicted_health == "Needs Attention" else "#e53935"

    # --- Crop Health Status Card ---
    st.markdown(
        f"""
        <div style="background:#36506c; border-radius:18px; padding:1rem; margin-bottom:0.8rem;">
            <span style="font-size:1.6rem;">🌱 <b>Crop Health:</b> <span style="color:{health_color};">{predicted_health}</span></span>
        </div>
        """, unsafe_allow_html=True
    )

    # --- Story Card ---
    stage = current_row['stage']
    temperature = current_row['temperature']
    humidity = current_row['humidity']
    n = current_row['n']
    p = current_row['p']
    k = current_row['k']
    st_ = current_row['st']
    sh_ = current_row['sh']
    ph = current_row['ph']

    # Compose story text based on logic
    if stage == "Dormancy":
        story_txt = "🌱 The saffron is in dormancy stage. No irrigation or fertilization is needed. The soil parameters are optimal."
    elif predicted_health == "Healthy":
        story_txt = "🌿 The saffron plant is thriving in optimal conditions. No immediate actions are required. 😊"
    elif predicted_health == "Needs Attention":
        # finer message
        if not (IDEAL["humidity_min"] <= humidity <= IDEAL["humidity_max"]):
            story_txt = "💧 The soil moisture is below the optimal range. Please irrigate the saffron as recommended to maintain healthy growth."
        elif n < IDEAL["n_min"] or p < IDEAL["p_min"] or k < IDEAL["k_min"]:
            story_txt = "🌾 The saffron plant requires additional fertilization. Please add the required nutrients according to the recommendations."
        elif not (IDEAL["temperature_min"] <= temperature <= IDEAL["temperature_max"]):
            if temperature < IDEAL["temperature_min"]:
                story_txt = "🥶 The temperature is lower than recommended. Monitor the greenhouse temperature to prevent stress on the plant."
            else:
                story_txt = "🔥 The temperature is above the optimal range. Cooling measures may be necessary to protect the saffron."
        else:
            story_txt = "⚠️ The saffron plant needs attention. Several parameters are out of the optimal range. Immediate action is recommended."
    elif predicted_health == "At Risk":
        story_txt = "🚨 The saffron plant is at risk due to critical parameters (e.g., pH, temperature). Please act quickly to stabilize the environment!"
    else:
        story_txt = "🤔 Unable to determine plant story."

    st.markdown(
        f"""
        <div style="background:#36506c; border-radius:18px; padding:1.2rem; margin-bottom:1.2rem; color:#fff;">
            {story_txt}
        </div>
        """, unsafe_allow_html=True
    )

    # --- Soil Table ---
    soil_params = ["n", "p", "k", "st", "sh"]
    current_values = [float(current_row[param]) for param in soil_params]
    recommendations, status, reasons = [], [], []
    for param, value in zip(soil_params, current_values):
        if stage == "Dormancy":
            recommendations.append("No irrigation (Dormancy)" if param == "sh" else "Optimal")
            status.append("Good")
            reasons.append("")
        elif param == "n":
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
            if not (IDEAL["sh_min"] <= value <= IDEAL["sh_max"]):
                recommendations.append(f"Add water: {int(np.maximum(0, 100*(IDEAL['sh_min']-value)/IDEAL['sh_min']))} ml")
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

    # --- Smooth Temperature Chart ---
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

# End

