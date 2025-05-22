import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

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

# 🌱 Growth Stage logic
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

    # 📘 Plant Story
    st.subheader("📘 Plant Story")
    if filtered_df['temperature'].values[0] > 27:
        story = "🌡 It’s a hot day inside the greenhouse.\nTemperatures are slightly above ideal. Ensure proper ventilation to keep the plant stress-free. 🍃"
    elif filtered_df['sh'].values[0] < 35:
        story = "💧 The soil is a bit dry, and the saffron seems thirsty.\nA moderate round of irrigation is recommended to restore balance. 💦"
    elif filtered_df['n'].values[0] < 50:
        story = "🧪 The plant is showing signs of nutrient deficiency.\nLow nitrogen levels detected – consider a light dose of N-rich fertilizer to boost development. 🌱"
    elif not (5.5 <= filtered_df['ph'].values[0] <= 8.0):
        story = "⚠️ The soil pH is outside the ideal range.\nThis may limit nutrient absorption. Consider adjusting the pH to support root activity. 🪴"
    elif (15 <= filtered_df['temperature'].values[0] <= 25 and 40 <= filtered_df['humidity'].values[0] <= 60 and 18 <= filtered_df['st'].values[0] <= 22):
        story = "🌿 The saffron plant is thriving today! \nOptimal temperature and moisture levels are supporting strong growth. No action needed – just keep watching it flourish. 🌞"
    else:
        story = "🚨 The saffron plant is under stress.\nSeveral parameters (like humidity and soil nutrients) are below optimal levels. Immediate attention is advised. 🌾"
    st.info(story)

    # 📆 Growth Stage
    month = selected_date.month
    stage = get_growth_stage(month)
    st.subheader("🪴 Growth Stage")
    st.info(f"📌 Current Growth Stage: **{stage}**")

else:
    st.warning("⚠️ No data available for the selected time.")
