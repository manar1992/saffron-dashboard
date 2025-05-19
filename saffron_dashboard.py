
# 🎨 Traffic Light Indicator - HTML + CSS style
st.markdown("### 🚦 Plant Health Traffic Light")

# Define active light based on prediction
health_colors = {
    "Healthy": "green",
    "Needs Attention": "orange",
    "At Risk": "red"
}

active_color = health_colors.get(predicted_health, "gray")

# Render traffic light using styled divs
traffic_light_html = f"""
<style>
.traffic-container {{
    width: 70px;
    background-color: #333;
    border-radius: 15px;
    padding: 15px;
    margin: auto;
}}
.light {{
    width: 40px;
    height: 40px;
    margin: 10px auto;
    border-radius: 50%;
    background-color: #555;
    opacity: 0.3;
}}
.light.green {{
    background-color: green;
    opacity: {"1.0" if active_color == "green" else "0.3"};
}}
.light.orange {{
    background-color: orange;
    opacity: {"1.0" if active_color == "orange" else "0.3"};
}}
.light.red {{
    background-color: red;
    opacity: {"1.0" if active_color == "red" else "0.3"};
}}
</style>

<div class="traffic-container">
    <div class="light red"></div>
    <div class="light orange"></div>
    <div class="light green"></div>
</div>
"""

st.markdown(traffic_light_html, unsafe_allow_html=True)
