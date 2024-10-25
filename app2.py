import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load the trained Isolation Forest model and scaler
iso_forest = joblib.load("isolation_forest_model_1_5_2.pkl")
scaler = joblib.load("scaler_1_5_2.pkl")

# Feature columns used during training
features_to_use = [
    "Heart Rate (BPM)", "Blood Oxygen Level (SpO2)", "Body Temperature (°C)",
    "Balance/Accelerometer", "Posture/Gyroscope", "Blood Pressure Systolic (mmHg)",
    "Blood Pressure Diastolic (mmHg)", "Gait Speed (m/s)", "Sugar Level (mg/dL)"
]

# Number of patients
num_patients = 5

# Streamlit page configuration
st.set_page_config(page_title="Real-Time Health Monitoring", layout="wide")
st.title("Real-Time Patient Health Monitoring Dashboard")

# Initialize session state for patient data
if "patient_data" not in st.session_state:
    st.session_state.patient_data = [
        {
            "Heart Rate (BPM)": 0,
            "Blood Oxygen Level (SpO2)": 0,
            "Body Temperature (°C)": 0,
            "Balance/Accelerometer": 0,
            "Posture/Gyroscope": 0,
            "Blood Pressure Systolic (mmHg)": 0,
            "Blood Pressure Diastolic (mmHg)": 0,
            "Gait Speed (m/s)": 0,
            "Sugar Level (mg/dL)": 0,
            "Anomaly": 0,
            "Fall Indicator": 0
        } for _ in range(num_patients)
    ]

# Patient-specific normal ranges
individual_ranges = [
    {
        "Heart Rate (BPM)": (60, 80),
        "Blood Oxygen Level (SpO2)": (96, 98),
        "Body Temperature (°C)": (36.7, 37.1),
        "Balance/Accelerometer": (0, 1),
        "Posture/Gyroscope": (0, 1),
        "Blood Pressure Systolic (mmHg)": (100, 120),
        "Blood Pressure Diastolic (mmHg)": (65, 80),
        "Gait Speed (m/s)": (0.9, 1.1),
        "Sugar Level (mg/dL)": (80, 120)
    },
    {
        "Heart Rate (BPM)": (70, 90),
        "Blood Oxygen Level (SpO2)": (95, 97),
        "Body Temperature (°C)": (36.6, 37.2),
        "Balance/Accelerometer": (0, 1),
        "Posture/Gyroscope": (0, 1),
        "Blood Pressure Systolic (mmHg)": (105, 125),
        "Blood Pressure Diastolic (mmHg)": (70, 85),
        "Gait Speed (m/s)": (0.8, 1.0),
        "Sugar Level (mg/dL)": (90, 130)
    },
    {
        "Heart Rate (BPM)": (65, 85),
        "Blood Oxygen Level (SpO2)": (97, 99),
        "Body Temperature (°C)": (36.5, 36.9),
        "Balance/Accelerometer": (0, 1),
        "Posture/Gyroscope": (0, 1),
        "Blood Pressure Systolic (mmHg)": (110, 130),
        "Blood Pressure Diastolic (mmHg)": (60, 75),
        "Gait Speed (m/s)": (1.0, 1.2),
        "Sugar Level (mg/dL)": (85, 115)
    },
    {
        "Heart Rate (BPM)": (75, 95),
        "Blood Oxygen Level (SpO2)": (94, 96),
        "Body Temperature (°C)": (36.8, 37.3),
        "Balance/Accelerometer": (0, 1),
        "Posture/Gyroscope": (0, 1),
        "Blood Pressure Systolic (mmHg)": (100, 115),
        "Blood Pressure Diastolic (mmHg)": (65, 80),
        "Gait Speed (m/s)": (0.7, 0.9),
        "Sugar Level (mg/dL)": (75, 110)
    },
    {
        "Heart Rate (BPM)": (68, 88),
        "Blood Oxygen Level (SpO2)": (95, 99),
        "Body Temperature (°C)": (36.6, 37.0),
        "Balance/Accelerometer": (0, 1),
        "Posture/Gyroscope": (0, 1),
        "Blood Pressure Systolic (mmHg)": (105, 120),
        "Blood Pressure Diastolic (mmHg)": (68, 78),
        "Gait Speed (m/s)": (0.85, 1.05),
        "Sugar Level (mg/dL)": (88, 108)
    }
]


# Function to generate normal data for a patient
def generate_normal_data(ranges):
    return {
        "Heart Rate (BPM)": np.random.uniform(*ranges["Heart Rate (BPM)"]),
        "Blood Oxygen Level (SpO2)": np.random.uniform(*ranges["Blood Oxygen Level (SpO2)"]),
        "Body Temperature (°C)": np.random.uniform(*ranges["Body Temperature (°C)"]),
        "Balance/Accelerometer": np.random.uniform(*ranges["Balance/Accelerometer"]),
        "Posture/Gyroscope": np.random.uniform(*ranges["Posture/Gyroscope"]),
        "Blood Pressure Systolic (mmHg)": np.random.uniform(*ranges["Blood Pressure Systolic (mmHg)"]),
        "Blood Pressure Diastolic (mmHg)": np.random.uniform(*ranges["Blood Pressure Diastolic (mmHg)"]),
        "Gait Speed (m/s)": np.random.uniform(*ranges["Gait Speed (m/s)"]),
        "Sugar Level (mg/dL)": np.random.uniform(*ranges["Sugar Level (mg/dL)"]),
        "Anomaly": 0,
        "Fall Indicator": 0
    }

# Function to display metrics for a patient
def display_patient_metrics(patient_id, data):
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.metric("Heart Rate", f"{data['Heart Rate (BPM)']:.1f} bpm", 
                  "Normal" if 60 <= data['Heart Rate (BPM)'] <= 100 else "Alert", 
                  delta_color="inverse")
        st.metric("Blood Pressure", 
                  f"{data['Blood Pressure Systolic (mmHg)']:.1f} / {data['Blood Pressure Diastolic (mmHg)']:.1f} mmHg",
                  "Normal" if (90 <= data['Blood Pressure Systolic (mmHg)'] <= 140) else "Elevated", 
                  delta_color="inverse")
        st.metric("SpO2", f"{data['Blood Oxygen Level (SpO2)']:.1f} %", 
                  "Normal" if data['Blood Oxygen Level (SpO2)'] >= 95 else "Low", 
                  delta_color="inverse")
    
    with col2:
        st.metric("Body Temperature", f"{data['Body Temperature (°C)']:.1f} °C",
                  "Normal" if 36 <= data['Body Temperature (°C)'] <= 37.5 else "High", 
                  delta_color="inverse")
        st.metric("Gait Speed", f"{data['Gait Speed (m/s)']:.2f} m/s", 
                  "Normal" if 0.5 <= data['Gait Speed (m/s)'] <= 1.2 else "Low", 
                  delta_color="inverse")
        st.metric("Sugar Level", f"{data['Sugar Level (mg/dL)']:.1f} mg/dL", 
                  "Normal" if 80 <= data['Sugar Level (mg/dL)'] <= 140 else "Alert", 
                  delta_color="inverse")
    
    with col3:
        status = "Normal" if data["Anomaly"] == 0 else "Anomaly"
        st.metric("Anomaly Status", status)
        fall_status = "Safe" if data["Fall Indicator"] == 0 else "Fall Risk"
        st.metric("Fall Risk", fall_status)

# Real-time monitoring for multiple patients
if st.button("Start Monitoring"):
    while True:
        for patient_id in range(num_patients):
            ranges = individual_ranges[patient_id]
            
            # Update the patient's data
            st.session_state.patient_data[patient_id] = generate_normal_data(ranges)

            # Update anomaly status (dummy logic, replace with real model prediction)
            st.session_state.patient_data[patient_id]["Anomaly"] = np.random.choice([0, 1], p=[0.95, 0.05])
            st.session_state.patient_data[patient_id]["Fall Indicator"] = np.random.choice([0, 1], p=[0.90, 0.10])

            st.subheader(f"Patient {patient_id + 1}")
            display_patient_metrics(patient_id, st.session_state.patient_data[patient_id])

        time.sleep(1)  # Adjust the delay as needed
