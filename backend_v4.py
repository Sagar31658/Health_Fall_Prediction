from flask import Flask, jsonify, Response, request
import pandas as pd
import numpy as np
import joblib
import json
import time
from flask_cors import CORS
import random
from asgiref.wsgi import WsgiToAsgi

app = Flask(__name__)
CORS(app)

iso_forest = joblib.load("isolation_forest_model_1_5_2.pkl")
scaler = joblib.load("scaler_1_5_2.pkl")

features_to_use = [
    "Heart Rate (BPM)", "Blood Oxygen Level (SpO2)", "Body Temperature (°C)",
    "Balance/Accelerometer", "Posture/Gyroscope", "Blood Pressure Systolic (mmHg)",
    "Blood Pressure Diastolic (mmHg)", "Gait Speed (m/s)", "Sugar Level (mg/dL)"
]

areas = {
    "Resident Room": {"X": (0, 20), "Y": (0, 20), "Z": (0, 2)},
    "Dining Area": {"X": (20, 40), "Y": (20, 40), "Z": (0, 2)},
    "Lounge Area": {"X": (40, 60), "Y": (40, 60), "Z": (0, 2)},
    "Garden/Yard": {"X": (60, 80), "Y": (60, 80), "Z": (0, 0)},
    "Nurse Station": {"X": (80, 100), "Y": (80, 100), "Z": (0, 2)},
    "Physical Therapy Room": {"X": (40, 60), "Y": (0, 20), "Z": (0, 2)},
    "Bathroom/Restroom": {"X": (20, 40), "Y": (0, 20), "Z": (0, 2)},
    "Library/Activity Room": {"X": (60, 80), "Y": (20, 40), "Z": (0, 2)},
    "Emergency Exit": {"X": (0, 20), "Y": (80, 100), "Z": (0, 0)},
    "Reception Area": {"X": (80, 100), "Y": (0, 20), "Z": (0, 2)}
}

patient_names = ["Alice Johnson", "Bob Smith", "Clara Davis", "David Wilson", "Emma Brown"]

# Individualized normal ranges for each patient
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

consecutive_anomaly_prob = 0.02  # Reduced anomaly probability
single_anomaly_prob = 0.03  # Reduced single anomaly probability
fall_trigger_prob = 0.05  # Reduced fall probability

# Global variable to control high-risk mode
high_risk_mode = False
fall_records = []

last_normal_values = [{} for _ in range(len(patient_names))]
 
def get_area(x, y, z):
    for area, coords in areas.items():
        if coords["X"][0] <= x <= coords["X"][1] and \
           coords["Y"][0] <= y <= coords["Y"][1] and \
           coords["Z"][0] <= z <= coords["Z"][1]:
            return area
    return random.choice(list(areas.keys()))

def classify_fall(row):
    gyro_threshold_forward = 20   # Threshold for forward fall (Gx)
    gyro_threshold_backward = -20 # Threshold for backward fall (Gx)
    gyro_threshold_left = 20      # Threshold for left side fall (Gy)
    gyro_threshold_right = -20    # Threshold for right side fall (Gy)

    if row['Gx'] > gyro_threshold_forward:
        return 'Fall Forward'
    elif row['Gx'] < gyro_threshold_backward:
        return 'Fall Backward'
    elif row['Gy'] > gyro_threshold_left:
        return 'Fall Left Side'
    elif row['Gy'] < gyro_threshold_right:
        return 'Fall Right Side'
    elif row['acc_mag'] > 15:  # Example threshold for fall detection based on acceleration magnitude
        return 'Fall Detected (Acceleration)'
    else:
        return 'Did Not Fall'


def generate_normal_data(ranges, patient_name, patient_id):
    global last_normal_values

    x = np.random.uniform(0, 100)
    y = np.random.uniform(0, 100)
    z = np.random.uniform(0, 2)
    area = get_area(x, y, z)

    Ax = np.random.uniform(-2, 2)  # Simulated accelerometer X-axis
    Ay = np.random.uniform(-2, 2)  # Simulated accelerometer Y-axis
    Az = np.random.uniform(-2, 2)  # Simulated accelerometer Z-axis
    Gx = np.random.uniform(-30, 30)  # Simulated gyroscope X-axis (fall forward/backward)
    Gy = np.random.uniform(-30, 30)  # Simulated gyroscope Y-axis (fall left/right)
    Gz = np.random.uniform(-30, 30)  # Simulated gyroscope Z-axis
    acc_mag = np.sqrt(Ax**2 + Ay**2 + Az**2)
    fall_direction = classify_fall({'Gx': Gx, 'Gy': Gy, 'acc_mag': acc_mag})



    if len(last_normal_values) <= patient_id:
        last_normal_values.extend([{}] * (patient_id - len(last_normal_values) + 1))

    # Generate fluctuating values close to the last normal value
    def get_fluctuated_value(feature, min_range, max_range, last_value):
        if last_value is None:
            return np.random.uniform(min_range, max_range)
        fluctuation = np.random.uniform(-0.5, 0.5)  # Small fluctuation range
        new_value = last_value + fluctuation
        return max(min(new_value, max_range), min_range)

    data = {
        "Patient Name": patient_name,
        "Heart Rate (BPM)": get_fluctuated_value(
            "Heart Rate (BPM)", *ranges["Heart Rate (BPM)"], last_normal_values[patient_id].get("Heart Rate (BPM)")
        ),
        "Blood Oxygen Level (SpO2)": get_fluctuated_value(
            "Blood Oxygen Level (SpO2)", *ranges["Blood Oxygen Level (SpO2)"], last_normal_values[patient_id].get("Blood Oxygen Level (SpO2)")
        ),
        "Body Temperature (°C)": get_fluctuated_value(
            "Body Temperature (°C)", *ranges["Body Temperature (°C)"], last_normal_values[patient_id].get("Body Temperature (°C)")
        ),
        "Balance/Accelerometer": np.random.uniform(*ranges["Balance/Accelerometer"]),
        "Posture/Gyroscope": np.random.uniform(*ranges["Posture/Gyroscope"]),
        "Blood Pressure Systolic (mmHg)": get_fluctuated_value(
            "Blood Pressure Systolic (mmHg)", *ranges["Blood Pressure Systolic (mmHg)"], last_normal_values[patient_id].get("Blood Pressure Systolic (mmHg)")
        ),
        "Blood Pressure Diastolic (mmHg)": get_fluctuated_value(
            "Blood Pressure Diastolic (mmHg)", *ranges["Blood Pressure Diastolic (mmHg)"], last_normal_values[patient_id].get("Blood Pressure Diastolic (mmHg)")
        ),
        "Gait Speed (m/s)": np.random.uniform(*ranges["Gait Speed (m/s)"]),
        "Sugar Level (mg/dL)": get_fluctuated_value(
            "Sugar Level (mg/dL)", *ranges["Sugar Level (mg/dL)"], last_normal_values[patient_id].get("Sugar Level (mg/dL)")
        ),
        "X coords": x,
        "Y coords": y,
        "Z": z,
        "Ax": Ax,
        "Ay": Ay,
        "Az": Az,
        "Gx": Gx,
        "Gy": Gy,
        "Gz": Gz,
        "Area": area,
        "Anomaly": 0,
        "Fall Indicator": 0,
        "Fall Direction": fall_direction
    }
    last_normal_values[patient_id] = {feature: data[feature] for feature in features_to_use}
    return data

def generate_anomaly_data(patient_name):
    x = np.random.uniform(0, 100)
    y = np.random.uniform(0, 100)
    z = np.random.uniform(0, 2)

    Ax = np.random.uniform(2, 5)  # Accelerometer X-axis
    Ay = np.random.uniform(2, 5)  # Accelerometer Y-axis
    Az = np.random.uniform(2, 5)  # Accelerometer Z-axis
    Gx = np.random.uniform(-40, 40)  # Gyroscope X-axis (fall forward/backward)
    Gy = np.random.uniform(-40, 40)  # Gyroscope Y-axis (fall left/right)
    Gz = np.random.uniform(-40, 40)  # Gyroscope Z-axis
    acc_mag = np.sqrt(Ax**2 + Ay**2 + Az**2)

    fall_direction = classify_fall({'Gx': Gx, 'Gy': Gy, 'acc_mag': acc_mag})

    area = get_area(x, y, z)
    return {
        "Patient Name": patient_name,
        "Heart Rate (BPM)": np.random.choice([np.random.uniform(40, 50), np.random.uniform(120, 150)]),
        "Blood Oxygen Level (SpO2)": np.random.uniform(85, 89),
        "Body Temperature (°C)": np.random.choice([np.random.uniform(34, 35), np.random.uniform(38.5, 39.5)]),
        "Balance/Accelerometer": np.random.uniform(2, 3),
        "Posture/Gyroscope": np.random.uniform(2, 3),
        "Blood Pressure Systolic (mmHg)": np.random.choice([np.random.uniform(70, 80), np.random.uniform(140, 160)]),
        "Blood Pressure Diastolic (mmHg)": np.random.choice([np.random.uniform(40, 50), np.random.uniform(100, 120)]),
        "Gait Speed (m/s)": np.random.uniform(0.2, 0.4),
        "Sugar Level (mg/dL)": np.random.choice([np.random.uniform(50, 60), np.random.uniform(200, 250)]),
        "X coords": x,
        "Y coords": y,
        "Z": z,
        "Area": area,
        "Ax": Ax,
        "Ay": Ay,
        "Az": Az,
        "Gx": Gx,
        "Gy": Gy,
        "Gz": Gz,
        "Anomaly": 1,
        "Fall Indicator": 0,
        "Fall Direction": fall_direction
    }


def detect_anomaly(data):
    df = pd.DataFrame([data])[features_to_use]
    X_scaled = scaler.transform(df)
    anomaly_score = iso_forest.predict(X_scaled)
    is_anomaly = int(anomaly_score[0] == -1)
    return is_anomaly

def stream_data_for_all_patients():
    global high_risk_mode,fall_records

    anomaly_block = 0
    fall_risk_counter = 0

    while True:
        all_patient_data = []

        for patient_id, (patient_name, ranges) in enumerate(zip(patient_names, individual_ranges), start=1):
            if high_risk_mode:
                # In high-risk mode, generate continuous anomalies
                data = generate_anomaly_data(patient_name)
                fall_risk_counter += 1
            else:
                # Handle normal anomaly generation with consecutive blocks
                if anomaly_block > 0:
                    data = generate_anomaly_data(patient_name)
                    anomaly_block -= 1
                    fall_risk_counter += 1
                elif np.random.rand() < consecutive_anomaly_prob:
                    anomaly_block = 9  # Start a block of 9 anomalies
                    data = generate_anomaly_data(patient_name)
                    fall_risk_counter += 1
                elif np.random.rand() < single_anomaly_prob:
                    data = generate_anomaly_data(patient_name)
                    fall_risk_counter = 0
                else:
                    data = generate_normal_data(ranges, patient_name, patient_id)
                    fall_risk_counter = 0

            # Detect anomaly using Isolation Forest
            data["Anomaly"] = detect_anomaly(data)

            # Evaluate fall risk based on gait, balance, and posture
            gait_balance_posture_risk = (
                data["Balance/Accelerometer"] > 2 or 
                data["Posture/Gyroscope"] > 2 or 
                data["Gait Speed (m/s)"] < 0.5
            )

            if gait_balance_posture_risk and np.random.rand() < 0.5:
                data["Fall Indicator"] = 1
                fall_risk_counter = 0
            elif fall_risk_counter >= 5 and np.random.rand() < fall_trigger_prob:
                data["Fall Indicator"] = 1
                fall_risk_counter = 0
            else:
                data["Fall Indicator"] = 0

            # Add patient ID and timestamp
            data["Patient ID"] = patient_id
            data["Timestamp"] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            if data["Fall Indicator"] == 1:
                fall_records.append(data)

            all_patient_data.append(data)

        # Yield the block of data for all patients together
        yield f"data:{json.dumps(all_patient_data)}\n\n"
        time.sleep(1)

@app.route('/')
def index():
    return "Hello, ASGI World!"

def stream_fall_records():
    global fall_records

    last_sent_index = 0  # Track the last sent fall record index

    while True:
        # Check if there are new fall records to send
        if len(fall_records) > last_sent_index:
            # Get the new fall records
            new_fall_records = fall_records[last_sent_index:]
            last_sent_index = len(fall_records)

            # Stream the new fall records
            yield f"data:{json.dumps(new_fall_records)}\n\n"
        
        time.sleep(1)

@app.route('/api/fall_records', methods=['GET'])
def fall_records_stream():
    """
    API route to stream fall records in real-time using SSE.
    """
    return Response(stream_fall_records(), content_type='text/event-stream')

@app.route('/api/data_stream', methods=['GET'])
def data_stream():
    return Response(stream_data_for_all_patients(), content_type='text/event-stream')

@app.route('/toggle_high_risk', methods=['POST'])
def toggle_high_risk():
    global high_risk_mode
    data = request.get_json()

    if 'enable' in data:
        high_risk_mode = data['enable']
        status = "enabled" if high_risk_mode else "disabled"
        return jsonify({"message": f"High-risk mode {status}."}), 200
    else:
        return jsonify({"error": "Missing 'enable' parameter."}), 400
    
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)