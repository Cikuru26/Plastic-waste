import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("smart_plastic_sensor_svm.joblib")

st.title("♻️ Smart Plastic Waste Detection (Sensor-based SVM)")
st.write("Enter sensor readings to predict whether the waste is **Plastic** or **Non-Plastic**")

# Input fields for 6 sensors
sensors = []
cols = st.columns(3)  # organize inputs in 2 rows
for i in range(6):
    with cols[i % 3]:
        val = st.number_input(f"Sensor {i+1}", value=0.0, format="%.3f")
        sensors.append(val)

# Prediction button
if st.button("Predict"):
    sample = np.array([sensors])
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    st.subheader("Result")
    if pred == 1:
        st.success(f"🟢 Plastic detected (Probability={prob:.2f})")
    else:
        st.info(f"🔵 Non-Plastic detected (Probability={1-prob:.2f})")
