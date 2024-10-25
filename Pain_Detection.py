import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (adjust the path as needed)
file_path = './imu_data_20240818_110636.csv'  # Replace with actual path
imu_data = pd.read_csv(file_path)

# Extract relevant features for model training
X = imu_data[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]

# Define thresholds for labeling falls
acc_threshold = 1.5  # Acceleration threshold
gyro_threshold = 1.5  # Gyroscope threshold

# Create synthetic labels based on thresholds
imu_data['Fall_Direction'] = 'No Fall'
imu_data.loc[(imu_data['Ax'] > acc_threshold) | (imu_data['Gx'] > gyro_threshold), 'Fall_Direction'] = 'Right'
imu_data.loc[(imu_data['Ax'] < -acc_threshold) | (imu_data['Gx'] < -gyro_threshold), 'Fall_Direction'] = 'Left'
imu_data.loc[(imu_data['Ay'] > acc_threshold) | (imu_data['Gy'] > gyro_threshold), 'Fall_Direction'] = 'Forward'
imu_data.loc[(imu_data['Ay'] < -acc_threshold) | (imu_data['Gy'] < -gyro_threshold), 'Fall_Direction'] = 'Backward'

# Extract labels
y = imu_data['Fall_Direction']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a Random Forest Classifier
fall_model = RandomForestClassifier(n_estimators=100, random_state=42)
fall_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = fall_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# Load new dataset (replace with actual file path)
new_file_path = './imu_data_20240818_110636.csv'  # Replace with actual path
new_imu_data = pd.read_csv(new_file_path)

# Extract relevant features
X_new_imu_data = new_imu_data[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]

# Make predictions
new_predictions = fall_model.predict(X_new_imu_data)

# Add predictions to the new dataset
new_imu_data['Predicted_Fall_Direction'] = new_predictions

# Display a few rows of the results
print(new_imu_data.head(20))

# Generate synthetic labels for the new dataset
new_imu_data['Synthetic_Fall_Direction'] = 'No Fall'
new_imu_data.loc[(new_imu_data['Ax'] > acc_threshold) | (new_imu_data['Gx'] > gyro_threshold), 'Synthetic_Fall_Direction'] = 'Right'
new_imu_data.loc[(new_imu_data['Ax'] < -acc_threshold) | (new_imu_data['Gx'] < -gyro_threshold), 'Synthetic_Fall_Direction'] = 'Left'
new_imu_data.loc[(new_imu_data['Ay'] > acc_threshold) | (new_imu_data['Gy'] > gyro_threshold), 'Synthetic_Fall_Direction'] = 'Forward'
new_imu_data.loc[(new_imu_data['Ay'] < -acc_threshold) | (new_imu_data['Gy'] < -gyro_threshold), 'Synthetic_Fall_Direction'] = 'Backward'

# Compare predicted vs synthetic labels
y_true = new_imu_data['Synthetic_Fall_Direction']
y_pred = new_imu_data['Predicted_Fall_Direction']

# Calculate accuracy and report
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
