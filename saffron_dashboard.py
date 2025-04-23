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
file_path = "green_house_saffron_1.csv"
if not os.path.exists(file_path):
    st.error(f"🚨 File '{file_path}' not found. Please upload it to the correct directory.")
    st.stop()

# 📥 Read data
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

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

# 🌿 Streamlit UI
st.title("🌱 Saffron Cultivation Dashboard")

# 📅 Select date
selected_date = st.sidebar.date_input("📅 Select Date", df['date'].min())

# 🕒 Select hour
time_slider = st.slider("⏰ Select Time:", 0, 23, step=1, format="%d:00")
filtered_df = df[(df['date'].dt.date == selected_date) & (df['time'].astype(str).str.startswith(str(time_slider).zfill(2)))]

# ✅ Display data if available
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡 Temperature", f"{filtered_df['temperature'].values[0]} °C")
    col2.metric("💧 Humidity", f"{filtered_df['humidity'].values[0]} %")
    col3.metric("🌤 Relative Humidity", f"{filtered_df['relative_humidity'].values[0]} %")

    # 🌱 Crop Health Status
    input_data = filtered_df[features].values[0]
    predicted_health = predict_crop_health(input_data)
    st.subheader("🌱 Crop Health Status")
    if predicted_health == "Healthy":
        st.success(f"🟢 Crop Health: {predicted_health}")
    else:
        st.error(f"🔴 Crop Health: {predicted_health}")

    # ⚠️ Alerts
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
    soil_data = {
        "Parameter": soil_params,
        "Current Value": [filtered_df[param].values[0] for param in soil_params],
        "Status": ["Bad" if param in ["n", "p", "k", "st", "sh", "ph"] else "Good" for param in soil_params],
        "Water Need": ["Sufficient Water"] * len(soil_params),
    }

    soil_df = pd.DataFrame(soil_data)

    recommendations = []
    for param, value in zip(soil_df["Parameter"], soil_df["Current Value"]):
        if param == "n":
            if value < 50:
                recommendations.append("Add Nitrogen: approx. 20 units")
            else:
                recommendations.append("No addition needed")
        elif param == "p":
            if not (0 <= value <= 1999):
                recommendations.append("Add Phosphorus: approx. 30 units")
            else:
                recommendations.append("No addition needed")
        elif param == "k":
            if not (0 <= value <= 1999):
                recommendations.append("Add Potassium: approx. 25 units")
            else:
                recommendations.append("No addition needed")
        else:
            recommendations.append("—")

    soil_df["Recommendation"] = recommendations
    soil_df = soil_df[["Parameter", "Current Value", "Recommendation", "Status", "Water Need"]]
    st.table(soil_df)

    # 📈 Temperature chart
    st.subheader("📈 Temperature Trend")
    temp_chart = px.line(df[df['date'].dt.date == selected_date], x="time", y="temperature", title="Temperature Over Time")
    st.plotly_chart(temp_chart)

else:
    st.warning("⚠️ No data available for the selected time.")
