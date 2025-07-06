import serial
import joblib
import xgboost as xgb
import numpy as np
import time
import requests

# Load trained models
loaded_xgb = xgb.Booster()
loaded_xgb.load_model("C:/Users/pvpra/OneDrive/Desktop/proj/xgb_model.json")

loaded_rf = joblib.load("C:/Users/pvpra/OneDrive/Desktop/proj/rf_model.pkl")
loaded_scaler = joblib.load("C:/Users/pvpra/OneDrive/Desktop/proj/scaler.pkl")

# OpenWeather API for temperature
def get_air_temperature():
    API_KEY = "407f11b45ba9f664441684afa58a198d"  # Replace with your API key
    CITY = "Coimbatore"
    URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

    response = requests.get(URL)
    data = response.json()
    if "main" in data:
        return data["main"]["temp"] + 273.15  # Convert Celsius to Kelvin
    return None

# Connect to Arduino
arduino = serial.Serial("COM9", 9600, timeout=1)  # Change COM port if needed
time.sleep(2)  # Allow connection time

print("✅ Connected to Arduino!")

start_time = time.time()
failure_count = 0  # Track consecutive failures
MAX_FAILURES = 5  # Stop after 5 consecutive failures

while True:
    try:
        # Read Data from Arduino
        arduino_data = arduino.readline().decode().strip()

        if "RPM:" in arduino_data:
            rpm = int(arduino_data.split(":")[1])  # Extract only RPM value
            print(f"⚙ RPM: {rpm}")

            # Get temperature from OpenWeather
            air_temp = get_air_temperature()
            process_temp = air_temp + 5  

            # Estimate Tool Wear
            tool_wear = (time.time() - start_time) / 60  

            # Assume Torque = 10 Nm
            torque = 10  

            print(f"🌡 Air Temp: {air_temp} K, 🔥 Process Temp: {process_temp} K, ⚙ RPM: {rpm}, 🔩 Torque: {torque} Nm, ⏳ Tool Wear: {tool_wear} min")

            # Prepare data for prediction
            new_motor_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
            new_motor_data_scaled = loaded_scaler.transform(new_motor_data)
            dnew = xgb.DMatrix(new_motor_data_scaled)

            # Predict Failure
            xgb_prediction = loaded_xgb.predict(dnew)
            rf_prediction = loaded_rf.predict(new_motor_data_scaled)

            xgb_prediction = (xgb_prediction > 0.5).astype(int)
            failure_detected = xgb_prediction[0] == 1 or rf_prediction[0] == 1

            if failure_detected:
                failure_count += 1
                print(f"⚠ FAILURE DETECTED! (Count: {failure_count}/{MAX_FAILURES})")
                arduino.write(b"FAILURE\n")

                # Stop monitoring after MAX_FAILURES
                if failure_count >= MAX_FAILURES:
                    print("❌ Motor has failed too many times! Stopping system.")
                    break  # Exit the loop

            else:
                failure_count = 0  # Reset failure count if motor is safe
                print("✅ Motor is SAFE")
                arduino.write(b"SAFE\n")

            time.sleep(2)  # Small delay to prevent excessive looping

    except Exception as e:
        print("❌ Error:",e)
