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

# 🟢 المثالي حسب الجداول المرفقة
IDEAL = {
    "ph_min": 6.0,
    "ph_max": 8.0,
    "temperature_min": 15,
    "temperature_max": 25,
    "humidity_min": 40,
    "humidity_max": 60,
    "n_min": 20,    # الحد الأدنى قبل الزراعة (أو في النمو المبكر)
    "n_max": 60,    # أعلى حد في النمو المتأخر
    "p_min": 60,
    "p_max": 80,
    "k_min": 40,
    "k_max": 60,
}

# 🟢 Crop health classification
def classify_crop_health(row):
    if not (IDEAL["ph_min"] <= row['ph'] <= IDEAL["ph_max"]):
        return "At Risk"
    elif not (IDEAL["temperature_min"] <= row['temperature'] <= IDEAL["temperature_max"]):
        return "Needs Attention"
    elif not (IDEAL["humidity_min"] <= row['humidity'] <= IDEAL["humidity_max"]):
        return "Needs Attention"
    # في الجدول مثالي النيتروجين والفوسفور والبوتاسيوم بالهكتار؛ لكن نستخدمها فقط لو تحت النطاق الأدنى للتنبيه
    elif row['n'] < IDEAL["n_min"]:
        return "Needs Attention"
    elif row['p'] < IDEAL["p_min"]:
        return "Needs Attention"
    elif row['k'] < IDEAL["k_min"]:
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

selected_date = st.sidebar.date_input("📅 Select Date", df['date_only'].min())
time_slider = st.slider("⏰ Select Hour:", 0, 23, step=1)
filtered_df = df[(df['date_only'] == selected_date) & (df['hour'] == time_slider)]

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

    # 📆 Growth Stage
    stage = filtered_df['stage'].values[0] if 'stage' in filtered_df.columns else "Unknown"
    st.subheader("🪴 Growth Stage")
    st.info(f"📌 Current Growth Stage: **{stage}**")

    # ⚠️ Alerts & Recommendations
    st.subheader("⚠️ Alerts & Recommendations")
    if not (IDEAL["humidity_min"] <= filtered_df['humidity'].values[0] <= IDEAL["humidity_max"]):
        st.warning("🚨 Humidity out of the ideal range (40–60%). Adjust irrigation as needed.")
    if not (IDEAL["temperature_min"] <= filtered_df['temperature'].values[0] <= IDEAL["temperature_max"]):
        st.warning("🌡️ Temperature out of the ideal range (15–25°C).")
    if not (IDEAL["ph_min"] <= filtered_df['ph'].values[0] <= IDEAL["ph_max"]):
        st.warning("🧪 pH out of the ideal range (6.0–8.0).")
    if filtered_df['n'].values[0] < IDEAL["n_min"]:
        st.error("⚠️ Nitrogen is below the ideal range (20–60 kg/ha).")
    if filtered_df['p'].values[0] < IDEAL["p_min"]:
        st.error("⚠️ Phosphorus is below the ideal range (60–80 kg/ha).")
    if filtered_df['k'].values[0] < IDEAL["k_min"]:
        st.error("⚠️ Potassium is below the ideal range (40–60 kg/ha).")

    # 🪴 Soil Details
    st.subheader("🪴 Soil Details")
    soil_params = ["n", "p", "k", "st", "sh", "ph"]
    current_values = [float(filtered_df[param].values[0]) for param in soil_params]

    recommendations = []
    status = []
    reasons = []
    for param, value in zip(soil_params, current_values):
        if param == "n":
            if value < IDEAL["n_min"]:
                recommendations.append("Add Nitrogen to reach at least 20 kg/ha")
                status.append("Bad")
                reasons.append("Low nitrogen")
            elif value > IDEAL["n_max"]:
                recommendations.append("Reduce Nitrogen application")
                status.append("Check")
                reasons.append("High nitrogen")
            else:
                recommendations.append("Optimal")
                status.append("Good")
                reasons.append("")
        elif param == "p":
            if value < IDEAL["p_min"]:
                recommendations.append("Add Phosphorus to reach at least 60 kg/ha")
                status.append("Bad")
                reasons.append("Low phosphorus")
            elif value > IDEAL["p_max"]:
                recommendations.append("Reduce Phosphorus application")
                status.append("Check")
                reasons.append("High phosphorus")
            else:
                recommendations.append("Optimal")
                status.append("Good")
                reasons.append("")
        elif param == "k":
            if value < IDEAL["k_min"]:
                recommendations.append("Add Potassium to reach at least 40 kg/ha")
                status.append("Bad")
                reasons.append("Low potassium")
            elif value > IDEAL["k_max"]:
                recommendations.append("Reduce Potassium application")
                status.append("Check")
                reasons.append("High potassium")
            else:
                recommendations.append("Optimal")
                status.append("Good")
                reasons.append("")
        elif param == "ph":
            if not (IDEAL["ph_min"] <= value <= IDEAL["ph_max"]):
                recommendations.append("Adjust pH to 6.0–8.0")
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
                reasons.append("Soil temp out of general range")
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

    st.table(soil_df)

    # 📈 Temperature chart
    st.subheader("📈 Temperature Trend")
    temp_chart = px.line(df[df['date_only'] == selected_date], x="hour", y="temperature", title="Temperature Over Time")
    st.plotly_chart(temp_chart)

else:
    st.warning("⚠️ No data available for the selected time.")

