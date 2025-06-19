import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from scipy.ndimage import gaussian_filter1d

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
        <div style="{card_style()} text-align:center; display:flex; flex-direction:column; align-items:center; min-width:160px;">
            <div style="font-size:2.2rem;">{icon}</div>
            <div style="font-size:1.7rem; font-weight:bold;">{value} <span style="font-size:1.1rem; color:#bbb;">{unit}</span></div>
            <div style="font-size:1.05rem; margin-top:0.15rem; color:#F7CA70;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Data Loading ----------
file_path = "saffron_greenhouse_synthetic_2years.csv"
if not os.path.exists(file_path):
    st.error(f"🚨 File '{file_path}' not found. Please upload it.")
    st.stop()

df = pd.read_csv(file_path)
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
}

def classify_crop_health(row):
    if not (IDEAL["ph_min"] <= row['ph'] <= IDEAL["ph_max"]):
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

df['crop_health'] = df.apply(classify_crop_health, axis=1)
label_encoder = LabelEncoder()
df['crop_health_label'] = label_encoder.fit_transform(df['crop_health'])
features = ["temperature", "humidity", "st", "ph", "n", "p", "k"]
X = df[features]
y = df["crop_health_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "crop_health_model.pkl")
loaded_model = joblib.load("crop_health_model.pkl")

def predict_crop_health(input_data):
    try:
        prediction = loaded_model.predict([input_data])[0]
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("<h2 style='color:#FFA500;'>🌱 Saffron Dashboard</h2>", unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", df['date_only'].min())
    time_slider = st.slider("🕒 Select Hour:", 0, 23, step=1)
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

# --- Quick Stats ---
if not filtered_df.empty:
    st.markdown("<div style='display:flex; gap:32px;'>", unsafe_allow_html=True)
    stat_card("🌡️", "Temperature", f"{filtered_df['temperature'].values[0]:.2f}", "°C")
    stat_card("💧", "Humidity", f"{filtered_df['humidity'].values[0]:.2f}", "%")
    stat_card("🧪", "pH", f"{filtered_df['ph'].values[0]:.2f}", "")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No data available for the selected time.")

# --- Growth Stage & Health Cards ---
if not filtered_df.empty:
    # Cards layout (Growth Stage, Health)
    colA, colB = st.columns([1, 2])
    with colA:
        st.markdown(
            f"""
            <div style="{card_style()}background:#283544;">
                <span style="font-size:1.25rem;">🪴 <b>Growth Stage</b></span><br>
                <div style="margin-top:0.55rem;">
                    <span style="font-size:1.13rem; color:#B5C5DF;">📌 Current Growth Stage: <b>{filtered_df['stage'].values[0]}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with colB:
        input_data = filtered_df[features].values[0]
        predicted_health = predict_crop_health(input_data)
        health_color = "#4CAF50" if predicted_health == "Healthy" else "#ff9800" if predicted_health == "Needs Attention" else "#e53935"
        st.markdown(
            f"""
            <div style="{card_style()}background:#232c2d;">
                <span style="font-size:1.25rem;">🩺 <b>Crop Health Status</b></span><br>
                <div style="margin-top:0.5rem; font-size:1.13rem;">
                    <span style="color:{health_color}; font-weight:bold;">{predicted_health}</span>
                </div>
                <div style="margin-top:0.5rem;">
                    {"🌿 Optimal. No immediate action required." if predicted_health == "Healthy" else
                     "⚠️ Several parameters are below optimal. Immediate attention advised." if predicted_health == "Needs Attention" else
                     "❗ Critical! Act quickly to stabilize the environment."}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

# --- Alerts & Recommendations ---
if not filtered_df.empty:
    alerts = []
    if not (IDEAL["humidity_min"] <= filtered_df['humidity'].values[0] <= IDEAL["humidity_max"]):
        alerts.append("💧 <b>Humidity out of range!</b> Adjust irrigation (40–60%).")
    if not (IDEAL["temperature_min"] <= filtered_df['temperature'].values[0] <= IDEAL["temperature_max"]):
        alerts.append("🌡️ <b>Temperature out of range!</b> Optimal: 15–25°C.")
    if not (IDEAL["ph_min"] <= filtered_df['ph'].values[0] <= IDEAL["ph_max"]):
        alerts.append("🧪 <b>pH out of range!</b> (6.0–8.0).")
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
            """, unsafe_allow_html=True
        )

# --- Soil Details Table ---
if not filtered_df.empty:
    soil_params = ["n", "p", "k", "st", "sh", "ph"]
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
        elif param == "ph":
            if not (IDEAL["ph_min"] <= value <= IDEAL["ph_max"]):
                recommendations.append("Adjust to 6.0–8.0")
                status.append("Bad")
                reasons.append("pH out of range")
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
            if not (IDEAL["humidity_min"] <= value <= IDEAL["humidity_max"]):
                recommendations.append("Adjust soil humidity")
                status.append("Check")
                reasons.append("Soil humidity out of range")
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

# End dashboard

