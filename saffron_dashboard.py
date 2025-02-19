import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 🟢 تهيئة الصفحة
st.set_page_config(page_title="Saffron Dashboard", layout="wide")

# 📂 تحديد المسار الصحيح للملف
file_path = "green_house_saffron_1.csv"

# 🟠 تحميل البيانات والتحقق من وجود الملف
if not os.path.exists(file_path):
    st.error(f"🚨 الملف '{file_path}' غير موجود. تأكد من تحميله إلى المستودع الصحيح.")
    st.stop()

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

# 🟢 تصنيف صحة المحصول
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

# 🔄 تحويل التصنيفات إلى أرقام
label_encoder = LabelEncoder()
df['crop_health_label'] = label_encoder.fit_transform(df['crop_health'])

# 📊 اختيار الميزات المستهدفة
features = ["temperature", "humidity", "st", "ph", "n", "p", "k"]
X = df[features]
y = df["crop_health_label"]

# 🔥 تدريب النموذج
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 حفظ النموذج
joblib.dump(model, "crop_health_model.pkl")

# 🚀 تحميل النموذج
loaded_model = joblib.load("crop_health_model.pkl")

# 🔍 دالة التنبؤ بصحة المحصول
def predict_crop_health(input_data):
    try:
        prediction = loaded_model.predict([input_data])[0]
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        return f"❌ خطأ في التنبؤ: {str(e)}"

# 🌿 **واجهة Streamlit**
st.title("🌱 Saffron Cultivation Dashboard")

# 📅 **اختيار التاريخ**
selected_date = st.sidebar.date_input("📅 Select Date", df['date'].min())

# 🕒 **اختيار الوقت**
st.subheader("📊 Select Time:")
time_slider = st.slider("حدد الوقت:", 0, 23, step=1, format="%d:00")
filtered_df = df[(df['date'].dt.date == selected_date) & (df['time'].astype(str).str.startswith(str(time_slider).zfill(2)))]

# ✅ التحقق من توفر البيانات
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡 Temperature", f"{filtered_df['temperature'].values[0]} °C")
    col2.metric("💧 Humidity", f"{filtered_df['humidity'].values[0]} %")
    col3.metric("🌤 Relative Humidity", f"{filtered_df['relative_humidity'].values[0]} %")

    # 🌱 **التنبؤ بصحة المحصول**
    input_data = filtered_df[features].values[0]
    predicted_health = predict_crop_health(input_data)
    st.subheader("🌱 Crop Health Prediction")
    st.write(f"🟢 **Predicted Crop Health: {predicted_health}**")

    # 🪴 **تفاصيل التربة**
    st.subheader("🪴 Soil Details")
    soil_params = ["temperature", "humidity", "relative_humidity", "n", "p", "k", "st", "sh", "ph"]
    
    soil_data = {
        "Parameter": soil_params,
        "Value": [filtered_df[param].values[0] for param in soil_params],
        "Status": ["Good" if 15 <= filtered_df["temperature"].values[0] <= 25 else "Bad",
                   "Good" if 40 <= filtered_df["humidity"].values[0] <= 60 else "Bad",
                   "Good" if 40 <= filtered_df["relative_humidity"].values[0] <= 60 else "Bad",
                   "Bad", "Bad", "Bad", "Bad", "Bad", "Bad"],
        "Water Need": ["Sufficient Water"] * 9
    }

    soil_df = pd.DataFrame(soil_data)
    st.table(soil_df)

    # 📈 **رسم بياني لدرجة الحرارة عبر الوقت**
    st.subheader("📈 Temperature Trend")
    temp_chart = px.line(df[df['date'].dt.date == selected_date], x="time", y="temperature", title="Temperature Over Time")
    st.plotly_chart(temp_chart)

else:
    st.warning("⚠️ لا توجد بيانات متاحة للوقت المحدد.")

# ✅ **حذف `st.run()` لتجنب الخطأ**
if __name__ == "__main__":
    st.write("✅ تطبيق Streamlit جاهز!")
