import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset (Make sure the file path is correct)
file_path = "green_house_saffron_1.csv"
df = pd.read_csv(file_path)


# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sidebar for date selection
selected_date = st.sidebar.date_input("Select Date", df['date'].min())

# Filter data based on selected date
filtered_df = df[df['date'].dt.date == selected_date]

# Display Title
st.title("🌿 Saffron Cultivation Dashboard")

# Time Selection Slider
st.subheader("Select Time:")
time_slider = st.slider("", 0, 23, step=1, format="%d:00")
filtered_df = filtered_df[filtered_df['time'].str.startswith(str(time_slider).zfill(2))]

# Display Key Metrics
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡 Temperature", f"{filtered_df['temperature'].values[0]} °C")
    col2.metric("💧 Humidity", f"{filtered_df['humidity'].values[0]} %")
    col3.metric("🌤 Relative Humidity", f"{filtered_df['relative_humidity'].values[0]} %")

    # Soil Details Table
    st.subheader("🪴 Soil Details")
    soil_params = ["temperature", "humidity", "relative_humidity", "n", "p", "k", "sc", "st", "sh", "ph"]
    
    soil_data = {
        "Parameter": soil_params,
        "Value": [filtered_df[param].values[0] for param in soil_params],
        "Status": ["Good" if 15 <= filtered_df["temperature"].values[0] <= 25 else "Bad",
                   "Good" if 40 <= filtered_df["humidity"].values[0] <= 60 else "Bad",
                   "Good" if 40 <= filtered_df["relative_humidity"].values[0] <= 60 else "Bad",
                   "Bad", "Bad", "Bad", "Bad", "Bad", "Bad", "Bad"],
        "Water Need": ["Sufficient Water"] * 10
    }

    soil_df = pd.DataFrame(soil_data)
    st.table(soil_df)

    # Temperature Trend
    st.subheader("📈 Temperature Trend")
    temp_chart = px.line(df[df['date'].dt.date == selected_date], x="time", y="temperature", title="Temperature Over Time")
    st.plotly_chart(temp_chart)

else:
    st.warning("No data available for the selected time.")

