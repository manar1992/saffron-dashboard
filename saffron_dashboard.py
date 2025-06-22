import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objs as go

st.set_page_config(page_title="Saffron Dashboard", layout="wide", initial_sidebar_state="expanded")

# --------- Theme & Icons ---------
PRIMARY_COLOR = "#7B3F00"    # Saffron brown
SECONDARY_COLOR = "#FFD700"  # Saffron yellow
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
df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['date'])
df['date_only'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

# Ideal Ranges
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
            f"""<div style="background:#223; color:#FFD700; padding:0.32rem 0.95rem; border-radius:11px; margin-top:0.7rem; display:inline-block; font-size:1.3rem;">
            🌱 <b>Stage:</b> <span style='color:#FFD700'>{growth_stage}</span>
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

# --- Crop Health Status Card + Story Together ---
if not filtered_df.empty:
    # Extract main values
    t, h, ph = filtered_df['temperature'].values[0], filtered_df['humidity'].values[0], filtered_df['ph'].values[0]
    n, p, k = filtered_df['n'].values[0], filtered_df['p'].values[0], filtered_df['k'].values[0]
    st_val, sh_val = filtered_df['st'].values[0], filtered_df['sh'].values[0]
    growth_stage = filtered_df['stage'].values[0]
    # Crop Health: calculate status
    is_healthy = (
        IDEAL["ph_min"] <= ph <= IDEAL["ph_max"]
        and IDEAL["temperature_min"] <= t <= IDEAL["temperature_max"]
        and IDEAL["humidity_min"] <= h <= IDEAL["humidity_max"]
        and IDEAL["n_min"] <= n <= IDEAL["n_max"]
        and IDEAL["p_min"] <= p <= IDEAL["p_max"]
        and IDEAL["k_min"] <= k <= IDEAL["k_max"]
        and IDEAL["st_min"] <= st_val <= IDEAL["st_max"]
        and IDEAL["sh_min"] <= sh_val <= IDEAL["sh_max"]
    )
    # Special Dormancy logic
    dormancy = (growth_stage.strip().lower() == "dormancy" or "سبات" in growth_stage)
    crop_status = "Healthy" if is_healthy else "Needs Attention"
    status_color = "#43A047" if crop_status == "Healthy" else "#d32f2f"
    # قصة النبات/التوصية
    if dormancy:
        story = "🌱 The saffron is in dormancy stage. No irrigation or fertilization is needed. The soil parameters are optimal."
        crop_status = "Healthy"
        status_color = "#43A047"
    elif is_healthy:
        story = "🌿 The saffron plant is thriving in optimal conditions. No immediate actions are required. 😊"
    else:
        story = "⚠️ The saffron plant requires attention. Some soil or environmental parameters are outside the optimal range."
    # Show card
    st.markdown(
        f"""
        <div style="background:#325172; border-radius:14px; margin-bottom:1rem; margin-top:1rem; padding:1.1rem 1.6rem;">
            <span style="font-size:1.5rem;"><b>🌱 Crop Health: <span style='color:{status_color};'>{crop_status}</span></b></span>
            <br><span style='font-size:1.2rem; margin-top:0.4rem; display:block;'>{story}</span>
        </div>
        """, unsafe_allow_html=True
    )

# --- Soil Details Table (without pH) ---
if not filtered_df.empty:
    soil_params = ["n", "p", "k", "st", "sh"]
    current_values = [float(filtered_df[param].values[0]) for param in soil_params]
    growth_stage = filtered_df['stage'].values[0]
    dormancy = (growth_stage.strip().lower() == "dormancy" or "سبات" in growth_stage)
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
            if not (IDEAL["st_min"] <= value <= IDEAL["st_max"]):
                recommendations.append("Adjust soil temp")
                status.append("Check")
                reasons.append("Soil temp out of range")
            else:
                recommendations.append("Optimal")
                status.append("Good")
                reasons.append("")
        elif param == "sh":
            if dormancy:
                recommendations.append("No irrigation (Dormancy)")
                status.append("Good")
                reasons.append("")
            elif value < IDEAL["sh_min"]:
                # If there's an actual 'irrigation_amount' column in your data, show ml
                if "irrigation_amount" in filtered_df.columns:
                    water_amt = float(filtered_df["irrigation_amount"].values[0])
                    recommendations.append(f"Add water: {water_amt:.0f} ml")
                else:
                    recommendations.append("Add water")
                status.append("Needs Water")
                reasons.append("Soil moisture is low")
            elif value > IDEAL["sh_max"]:
                recommendations.append("Reduce irrigation")
                status.append("Check")
                reasons.append("Soil moisture is high")
            else:
                recommendations.append("Optimal")
                status.append("Good")
                reasons.append("")
    # Build dataframe for display
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
