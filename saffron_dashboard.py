# Full working Streamlit app including Soil Details with reasoning and formatting

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 🟢 Page config
st.set_page_config(page_title="Saffron Dashboard", layout="wide")

# 📂 Load dataset
file_path = "green_house_saffron_1.csv"
if not os.path.exists(file_path):
    st.error(f"🚨 File '{file_path}' not found.")
    st.stop()

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

# 🟢 Initial display time from first record
initial_date = df['date'].dt.date.iloc[0]
initial_hour = int(df['time'].astype(str).str[:2].iloc[0])

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
label_encoder = LabelEncoder()
df['crop_health_label'] = label_encoder.fit_transform(df['crop_health'])

features = ["temperature", "humidity", "st", "ph", "n", "p", "k"]
X = df[features]
y = df["crop_health_label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "crop_health_model.pkl")
loaded_model = joblib.load("crop_health_model.pkl")

def predict_crop_health(input_data):
    prediction = loaded_model.predict([input_data])[0]
    return label_encoder.inverse_transform([prediction])[0]

# 🌿 Streamlit UI
st.title("🌱 Saffron Cultivation Dashboard")

# Sidebar inputs
selected_date = st.sidebar.date_input("📅 Select Date", value=initial_date)
time_slider = st.slider("⏰ Select Time:", 0, 23, step=1, value=initial_hour, format="%d:00")
filtered_df = df[(df['date'].dt.date == selected_date) & (df['time'].astype(str).str.startswith(str(time_slider).zfill(2)))]

# Display content if data exists
if not filtered_df.empty:
    st.metric("🌡 Temperature", f"{filtered_df['temperature'].values[0]} °C")
    st.metric("💧 Humidity", f"{filtered_df['humidity'].values[0]} %")
    st.metric("☁️ Relative Humidity", f"{filtered_df['relative_humidity'].values[0]} %")

    input_data = filtered_df[features].values[0]
    predicted_health = predict_crop_health(input_data)

    st.subheader("🩺 Crop Health Status")
    if predicted_health == "Healthy":
        st.success(f"🟢 Crop Health: {predicted_health}")
    else:
        st.error(f"🔴 Crop Health: {predicted_health}")

    # 🪴 Soil Details
    st.subheader("🪴 Soil Details")
    soil_params = ["n", "p", "k", "st", "sh", "ph"]
    soil_data = {
        "Parameter": soil_params,
        "Current Value": [int(filtered_df[param].values[0]) for param in soil_params],
        "Recommendation": [],
        "Status": [],
        "Water Need": ["Sufficient Water"] * len(soil_params),
        "Reason": []
    }

    for param, value in zip(soil_data["Parameter"], soil_data["Current Value"]):
        rec = "—"
        status = "Good"
        reason = ""

        if param == "n":
            if value < 50:
                rec = "Add Nitrogen: approx. 20 units"
                status = "Bad"
                reason = "Low nitrogen level"
            elif value > 500:
                rec = "No addition needed"
                status = "Bad"
                reason = "Excess nitrogen level"
        elif param == "p":
            if not (0 <= value <= 1999):
                rec = "Adjust Phosphorus"
                status = "Bad"
                reason = "Phosphorus out of range"
            else:
                rec = "No addition needed"
                status = "Bad"
                reason = "Outside optimal range"
        elif param == "k":
            if not (0 <= value <= 1999):
                rec = "Adjust Potassium"
                status = "Bad"
                reason = "Potassium out of range"
            else:
                rec = "No addition needed"
                status = "Bad"
                reason = "Outside optimal range"
        elif param == "st":
            if not (18 <= value <= 22):
                status = "Bad"
                reason = "Soil temperature out of range"
        elif param == "sh":
            if not (40 <= value <= 60):
                status = "Bad"
                reason = "Soil humidity out of range"
        elif param == "ph":
            if not (5.5 <= value <= 8.0):
                status = "Bad"
                reason = "pH level out of range"

        soil_data["Recommendation"].append(rec)
        soil_data["Status"].append(status)
        soil_data["Reason"].append(reason)

    soil_df = pd.DataFrame(soil_data)
    st.dataframe(soil_df)
else:
    st.warning("⚠️ No data available for the selected time.")

