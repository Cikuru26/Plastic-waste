import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("smart_plastic_sensor_svm.joblib")

st.title("♻️ Smart Plastic Waste Detection (Sensor-based SVM)")
st.write("This system uses 6 sensors to classify waste as **Plastic** or **Non-Plastic**.")

# Show diagram of Smart Bin
st.image("smart_bin_diagram.png", caption="Smart Bin with 6 Sensors", use_container_width=True)

# Define sensor names
sensor_names = [
    "Infrared (IR) Sensor",
    "Capacitive Sensor",
    "Moisture Sensor",
    "Ultrasonic Sensor",
    "Color Sensor",
    "Weight Sensor (Load Cell)"
]

# Input fields for sensors
st.subheader("🔧 Enter Sensor Readings")
sensors = []
cols = st.columns(3)  # organize inputs in 2 rows
for i, name in enumerate(sensor_names):
    with cols[i % 3]:
        val = st.number_input(f"{name}", value=0.0, format="%.3f")
        sensors.append(val)

# Prediction button
if st.button("Predict"):
    sample = np.array([sensors])
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.success(f"🟢 Plastic detected (Probability={prob:.2f})")
    else:
        st.info(f"🔵 Non-Plastic detected (Probability={1-prob:.2f})")

