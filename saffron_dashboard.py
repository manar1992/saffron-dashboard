import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib
import os
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Saffron Dashboard", layout="wide", initial_sidebar_state="expanded")

# --------- Theme & Icons ---------
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
df['month'] = df['datetime'].dt.month

# ----------- Load ML Models ----------
try:
    clf_model = joblib.load('clf_model.pkl')
    reg_model = joblib.load('reg_model.pkl')
    model_fertilization_need = joblib.load('model_fertilization_need.pkl')
    model_fertilization_type = joblib.load('model_fertilization_type.pkl')
    model_fertilization_amount = joblib.load('model_fertilization_amount.pkl')
    fertilizer_label_encoder = joblib.load('label_encoder_type.pkl')
except Exception as e:
    st.error(f"🚨 Error loading one of the models: {e}")
    st.stop()

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("<h2 style='color:#FFA500;'>🌱 Saffron Dashboard</h2>", unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", df['date_only'].min())
    time_slider = st.slider("🕒 Select Hour:", 0, 23, step=1)

    # Show Growth Stage as a badge under date/time
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

# --- ML Prediction Section ---
if not filtered_df.empty:
    # Prepare feature vectors
    irrigation_features = ['temperature', 'humidity', 'st', 'sh', 'month']
    fert_features = ["temperature", "humidity", "ph", "sc", "sh", "st", "n", "p", "k", "month"]
    irrigation_input = filtered_df[irrigation_features].astype(float).values[0]
    fert_input = filtered_df[fert_features].astype(float).values[0]

    # Irrigation Prediction
    irrigation_need = clf_model.predict([irrigation_input])[0]
    if irrigation_need == 1:
        irrigation_amount = reg_model.predict([irrigation_input])[0]
    else:
        irrigation_amount = 0

    # Fertilization Prediction
    fert_need = model_fertilization_need.predict([fert_input])[0]
    if fert_need == 1:
        fert_type = model_fertilization_type.predict([fert_input])[0]
        fert_type_label = fertilizer_label_encoder.inverse_transform([fert_type])[0]
        fert_amount = model_fertilization_amount.predict([fert_input])[0]
    else:
        fert_type_label = "None"
        fert_amount = 0

    # --- Irrigation Card ---
    st.markdown(
        f"""
        <div style="{card_style()}background:#222;">
            <span style="font-size:1.25rem;">💧 <b>Irrigation Recommendation</b></span><br>
            <div style="margin-top:0.5rem; font-size:1.13rem;">
                <b>Needs Irrigation:</b> <span style="color:#FFD700;">{'Yes' if irrigation_need == 1 else 'No'}</span>
            </div>
            {"<div style='margin-top:0.5rem;'><b>Recommended Amount:</b> <span style='color:#4CAF50;'>{:.1f} ml/m²</span></div>".format(irrigation_amount) if irrigation_need == 1 else ""}
        </div>
        """, unsafe_allow_html=True
    )

    # --- Fertilization Card ---
    st.markdown(
        f"""
        <div style="{card_style()}background:#254161;">
            <span style="font-size:1.25rem;">🧪 <b>Fertilization Recommendation</b></span><br>
            <div style="margin-top:0.5rem; font-size:1.13rem;">
                <b>Needs Fertilization:</b> <span style="color:#FFD700;">{'Yes' if fert_need == 1 else 'No'}</span>
            </div>
            {f"<div style='margin-top:0.5rem;'><b>Type:</b> <span style='color:#4CAF50;'>{fert_type_label}</span></div>" if fert_need == 1 else ""}
            {f"<div style='margin-top:0.5rem;'><b>Amount:</b> <span style='color:#4CAF50;'>{fert_amount:.1f} kg/ha</span></div>" if fert_need == 1 else ""}
        </div>
        """, unsafe_allow_html=True
    )

# يمكنك هنا إبقاء جداول وتحليلات أخرى مثل soil details, charts...
# (لم أغير بقية الكود الخاص بالجداول أو المخططات حتى لا تزيد التعقيد)

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
