import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 🟢 Page configuration
st.set_page_config(page_title="Saffron Dashboard", layout="wide")

# 📂 Load dataset
file_path = "saffron_greenhouse_synthetic_2years.csv"
if not os.path.exists(file_path):
    st.error(f"🚨 File '{file_path}' not found. Please upload it to the correct directory.")
    st.stop()

# 📥 Read data
df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['date'])
df['date_only'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

# 🟢 Crop health classification
def classify_crop_health(row):
    if row['ph'] < 5.5 or row['ph'] > 8.0:
        return "At Risk"
    elif row['temperature'] < 15 or row['temperature'] > 25:
        return "Needs Attention"
    elif row['humidity'] < 40 or row['humidity'] > 60:
        return "Needs Attention"
    elif row['st'] < 18 or row['st'] > 22:
        return "Needs Attention"
    else:
        return "Healthy"

df['crop_health'] = df.apply(classify_crop_health, axis=1)

# 🔄 Encode labels
label_encoder = LabelEncoder()
df['crop_health_label'] = label_encoder.fit_transform(df['crop_health'])

# 📊 Feature selection
features = ["temperature", "humidity", "st", "ph", "n", "p", "k"]
X = df[features]
y = df["crop_health_label"]

# 🔥 Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Save model
joblib.dump(model, "crop_health_model.pkl")

# 🚀 Load model
loaded_model = joblib.load("crop_health_model.pkl")

# 🔍 Prediction function
def predict_crop_health(input_data):
    try:
        prediction = loaded_model.predict([input_data])[0]
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

# 🌱 Growth Stage logic
def get_growth_stage(stage_val):
    # stage موجود بالملف الجديد
    if pd.isnull(stage_val):
        return "Unknown"
    return stage_val

# 🌿 Streamlit UI
st.title("🌱 Saffron Cultivation Dashboard")

# 📅 Select date
selected_date = st.sidebar.date_input("📅 Select Date", df['date_only'].min())
# 🕒 Select hour
time_slider = st.slider("⏰ Select Hour:", 0, 23, step=1)

filtered_df = df[(df['date_only'] == selected_date) & (df['hour'] == time_slider)]

# ✅ Display data if available
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡 Temperature", f"{filtered_df['temperature'].values[0]:.2f} °C")
    col2.metric("💧 Humidity", f"{filtered_df['humidity'].values[0]:.2f} %")
    col3.metric("🧪 pH", f"{filtered_df['ph'].values[0]:.2f}")

    # 🌱 Crop Health Status
    input_data = filtered_df[features].values[0]
    predicted_health = predict_crop_health(input_data)
    st.subheader("🌱 Crop Health Status")
    if predicted_health == "Healthy":
        st.success(f"🟢 Crop Health: {predicted_health}")
    elif predicted_health == "Needs Attention":
        st.error(f"🔴 Crop Health: {predicted_health}")
    else:
        st.warning(f"🟠 Crop Health: {predicted_health}")

    # 📖 Plant Story
    if predicted_health == "Healthy":
        st.info("🌿 The saffron plant is thriving in optimal conditions. No immediate actions are required. 😊")
    elif predicted_health == "Needs Attention":
        st.info("🚨 The saffron plant is under stress. Several parameters (like humidity and soil nutrients) are below optimal levels. Immediate attention is advised. 🌾")
    elif predicted_health == "At Risk":
        st.warning("⚠️ The saffron plant is facing critical conditions. pH or temperature is far from the recommended range. Act quickly to stabilize the environment. ❗")
    else:
        st.info("🤔 Unable to determine plant story.")

    # 📆 Growth Stage
    stage = get_growth_stage(filtered_df['stage'].values[0]) if 'stage' in filtered_df.columns else "Unknown"
    st.subheader("🪴 Growth Stage")
    st.info(f"📌 Current Growth Stage: **{stage}**")

    # ⚠️ Alerts & Recommendations
    st.subheader("⚠️ Alerts & Recommendations")
    if filtered_df['humidity'].values[0] < 40 or filtered_df['st'].values[0] < 18:
        st.warning("🚨 Irrigation Needed: Humidity or soil moisture is below optimal level.")
    if filtered_df['n'].values[0] < 50:
        st.error("⚠️ Fertilizer Needed: Nitrogen is low.")
    if not (0 <= filtered_df['p'].values[0] <= 1999):
        st.error("⚠️ Fertilizer Needed: Phosphorus is out of range.")
    if not (0 <= filtered_df['k'].values[0] <= 1999):
        st.error("⚠️ Fertilizer Needed: Potassium is out of range.")

    # 🪴 Soil Details
    st.subheader("🪴 Soil Details")
    soil_params = ["n", "p", "k", "st", "sh", "ph"]
    current_values = [float(filtered_df[param].values[0]) for param in soil_params]

    recommendations = []
    status = []
    reasons = []
    for param, value in zip(soil_params, current_values):
        if param == "n":
            if value < 50:
                recommendations.append("Add Nitrogen: approx. 20 units")
                status.append("Bad")
                reasons.append("Low nitrogen")
            else:
                recommendations.append("No addition needed")
                status.append("Good")
                reasons.append("")
        elif param == "p":
            if not (0 <= value <= 1999):
                recommendations.append("Add Phosphorus: approx. 30 units")
                status.append("Bad")
                reasons.append("Phosphorus out of range")
            else:
                recommendations.append("No addition needed")
                status.append("Good")
                reasons.append("")
        elif param == "k":
            if not (0 <= value <= 1999):
                recommendations.append("Add Potassium: approx. 25 units")
                status.append("Bad")
                reasons.append("Potassium out of range")
            else:
                recommendations.append("No addition needed")
                status.append("Good")
                reasons.append("")
        elif param == "st":
            if not (18 <= value <= 22):
                recommendations.append("Adjust soil temp")
                status.append("Bad")
                reasons.append("Soil temp out of range")
            else:
                recommendations.append("—")
                status.append("Good")
                reasons.append("")
        elif param == "sh":
            if not (40 <= value <= 60):
                recommendations.append("Improve soil humidity")
                status.append("Bad")
                reasons.append("Soil humidity out of range")
            else:
                recommendations.append("—")
                status.append("Good")
                reasons.append("")
        elif param == "ph":
            if not (5.5 <= value <= 8.0):
                recommendations.append("Adjust pH level")
                status.append("Bad")
                reasons.append("pH out of range")
            else:
                recommendations.append("—")
                status.append("Good")
                reasons.append("")

    soil_df = pd.DataFrame({
        "Parameter": soil_params,
        "Current Value": current_values,
        "Recommendation": recommendations,
        "Status": status,
        "Reason": reasons,
        "Water Need": ["Sufficient Water"] * len(soil_params),
    })

    soil_df = soil_df[["Parameter", "Current Value", "Recommendation", "Status", "Reason", "Water Need"]]
    st.table(soil_df)

    # 📈 Temperature chart
    st.subheader("📈 Temperature Trend")
    temp_chart = px.line(df[df['date_only'] == selected_date], x="hour", y="temperature", title="Temperature Over Time")
    st.plotly_chart(temp_chart)

else:
    st.warning("⚠️ No data available for the selected time.")

