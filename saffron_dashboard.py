import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# Default initial timestamp values from first row
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

# 📅 Select date (defaults to first available date)
selected_date = st.sidebar.date_input("📅 Select Date", value=initial_date)

# 🕒 Select hour (defaults to hour of first row)
time_slider = st.slider("⏰ Select Time:", 0, 23, step=1, value=initial_hour, format="%d:00")
filtered_df = df[(df['date'].dt.date == selected_date) & (df['time'].astype(str).str.startswith(str(time_slider).zfill(2)))]

# Rest of the dashboard remains unchanged ...
