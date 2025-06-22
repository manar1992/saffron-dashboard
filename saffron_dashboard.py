import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Saffron Dashboard", layout="wide", initial_sidebar_state="expanded")

# --------- Theme & Icons ---------
PRIMARY_COLOR = "#FFA500"
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

# ---------- Data Loading ----------
file_path = "saffron_greenhouse_synthetic_2years.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"🚨 Error loading file '{file_path}': {e}")
    st.stop()

df['datetime'] = pd.to_datetime(df['date'])
df['date_only'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

# ------------- Ideal Ranges Per Stage -------------
IDEAL_RANGES = {
    'Dormancy': {
        "ph": (6.0, 8.0), "temperature": (15, 25), "humidity": (40, 60),
        "n": (20, 60), "p": (60, 80), "k": (40, 60), "st": (18, 22), "sh": (30, 70)
    },
    'Growth Stimulation': {
        "ph": (6.0, 8.0), "temperature": (15, 25), "humidity": (40, 60),
        "n": (20, 60), "p": (60, 80), "k": (40, 60), "st": (18, 22), "sh": (35, 70)
    },
    'Vegetative Growth': {
        "ph": (6.0, 8.0), "temperature": (15, 25), "humidity": (40, 60),
        "n": (40, 60), "p": (60, 80), "k": (40, 60), "st": (18, 22), "sh": (40, 70)
    },
    'Flowering': {
        "ph": (6.0, 8.0), "temperature": (15, 25), "humidity": (40, 60),
        "n": (30, 50), "p": (60, 80), "k": (30, 60), "st": (18, 22), "sh": (35, 70)
    },
    'Corm Multiplication': {
        "ph": (6.0, 8.0), "temperature": (15, 25), "humidity": (40, 60),
        "n": (20, 60), "p": (60, 80), "k": (40, 60), "st": (18, 22), "sh": (40, 70)
    },
    'Leaf Yellowing & Dormancy Preparation': {
        "ph": (6.0, 8.0), "temperature": (15, 25), "humidity": (40, 60),
        "n": (20, 60), "p": (60, 80), "k": (40, 60), "st": (18, 22), "sh": (30, 70)
    }
}

# Story templates by crop health
HEALTH_STORIES = {
    "Dormancy": "🌱 The saffron is in dormancy stage. No irrigation or fertilization is needed. The soil parameters are optimal.",
    "Healthy": "🌿 The saffron plant is thriving in optimal conditions. No immediate actions are required. 😊",
    "Needs Attention": "⚠️ The saffron plant needs attention. Some parameters are outside the ideal range. Check recommendations below.",
    "At Risk": "🚨 Critical risk! Multiple parameters are out of the optimal range. Please act quickly and check recommendations."
}

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("<h2 style='color:#FFA500;'>🌱 Saffron Dashboard</h2>", unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", df['date_only'].min())
    time_slider = st.slider("🕒 Select Hour:", 0, 23, step=1)

    # Show Growth Stage as a badge under date/time
    selected_row = df[(df['date_only'] == selected_date) & (df['hour'] == time_slider)]
    growth_stage = selected_row['stage'].values[0] if not selected_row.empty else "N/A"
    st.markdown(
        f"""<div style="background:#223; color:#FFD700; padding:0.32rem 0.95rem; border-radius:11px; margin-top:0.7rem; display:inline-block; font-size:1.17rem;">
        🌱 <b>Stage:</b> <span style="color:yellow;">{growth_stage}</span>
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

# --- Mini Cards (Temperature, Humidity, pH) in one row ---
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

# ---------- Crop Health Logic ----------
def assess_crop_health(row):
    stage = row['stage']
    stage_range = IDEAL_RANGES.get(stage, IDEAL_RANGES['Dormancy'])  # fallback to Dormancy
    problems = []

    # Ignore everything if Dormancy: always healthy!
    if stage == "Dormancy":
        return "Healthy", HEALTH_STORIES["Dormancy"], []
    # Check parameters
    if not (stage_range['ph'][0] <= row['ph'] <= stage_range['ph'][1]):
        problems.append("pH out of range")
    if not (stage_range['temperature'][0] <= row['temperature'] <= stage_range['temperature'][1]):
        problems.append("Temperature out of range")
    if not (stage_range['humidity'][0] <= row['humidity'] <= stage_range['humidity'][1]):
        problems.append("Humidity out of range")
    if not (stage_range['n'][0] <= row['n'] <= stage_range['n'][1]):
        problems.append("Nitrogen out of range")
    if not (stage_range['p'][0] <= row['p'] <= stage_range['p'][1]):
        problems.append("Phosphorus out of range")
    if not (stage_range['k'][0] <= row['k'] <= stage_range['k'][1]):
        problems.append("Potassium out of range")
    if not (stage_range['st'][0] <= row['st'] <= stage_range['st'][1]):
        problems.append("Soil temp out of range")
    if not (stage_range['sh'][0] <= row['sh'] <= stage_range['sh'][1]):
        problems.append("Soil moisture out of range")

    # Health decision
    if len(problems) == 0:
        return "Healthy", HEALTH_STORIES["Healthy"], []
    elif len(problems) <= 2:
        return "Needs Attention", HEALTH_STORIES["Needs Attention"], problems
    else:
        return "At Risk", HEALTH_STORIES["At Risk"], problems

if not filtered_df.empty:
    row = filtered_df.iloc[0]
    crop_health, story, problems = assess_crop_health(row)
    health_color = "#4CAF50" if crop_health == "Healthy" else "#ff9800" if crop_health == "Needs Attention" else "#e53935"
    st.markdown(
        f"""
        <div style="background:#38516B; border-radius:17px; padding:1.1rem 1.4rem 1.1rem 1.4rem; margin-bottom:0.6rem;">
            <span style="font-size:2rem;vertical-align:middle;">🌱</span>
            <span style="font-size:1.75rem;font-weight:800;">Crop Health: <span style="color:{health_color}">{crop_health}</span></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="background:#38516B; border-radius:17px; padding:1.2rem 1.4rem 0.5rem 1.4rem; margin-bottom:1.2rem; color:#fff;">
            <span style="font-size:1.35rem;vertical-align:middle;">🌱</span>
            <span style="font-size:1.17rem; font-weight:400;">{story}</span>
            {"<ul style='margin-top:0.7rem;'>" + "".join([f"<li style='font-size:1rem;color:#FFD700;'>{p}</li>" for p in problems]) + "</ul>" if problems else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Soil Details Table ---
if not filtered_df.empty:
    soil_params = ["n", "p", "k", "st", "sh"]
    stage = row['stage']
    stage_range = IDEAL_RANGES.get(stage, IDEAL_RANGES['Dormancy'])
    current_values = [float(row[param]) for param in soil_params]
    recommendations, status, reasons = [], [], []
    for param, value in zip(soil_params, current_values):
        rng = stage_range[param]
        if param == "sh" and stage == "Dormancy":
            recommendations.append("No irrigation (Dormancy)")
            status.append("Good")
            reasons.append("")
        elif value < rng[0]:
            if param == "n":
                recommendations.append(f"Add N to ≥{rng[0]} kg/ha")
                status.append("Needs N")
                reasons.append("Low nitrogen")
            elif param == "p":
                recommendations.append(f"Add P to ≥{rng[0]} kg/ha")
                status.append("Needs P")
                reasons.append("Low phosphorus")
            elif param == "k":
                recommendations.append(f"Add K to ≥{rng[0]} kg/ha")
                status.append("Needs K")
                reasons.append("Low potassium")
            elif param == "st":
                recommendations.append("Adjust soil temp")
                status.append("Check")
                reasons.append("Soil temp out of range")
            elif param == "sh":
                ml_amount = int(abs(value - rng[0]))  # تقريبًا كمية ماء مقترحة
                recommendations.append(f"Add water: {ml_amount} ml")
                status.append("Needs Water")
                reasons.append("Soil moisture is low")
        elif value > rng[1]:
            if param in ["n", "p", "k"]:
                recommendations.append(f"Reduce {param.upper()}")
                status.append("Check")
                reasons.append(f"High {param}")
            elif param == "st":
                recommendations.append("Adjust soil temp")
                status.append("Check")
                reasons.append("Soil temp out of range")
            elif param == "sh":
                recommendations.append("Reduce irrigation")
                status.append("Check")
                reasons.append("Soil moisture high")
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
if not filtered_df.empty:
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
