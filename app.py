# app.py
from flask import Flask, render_template, request
import requests
import joblib
import pandas as pd
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

API_KEY = "2eaf37170403bf8c6f9b32a12b525399"
BASE_URL = "https://api.openweathermap.org/data/2.5/"

# Load trained models
rain_model = joblib.load("models/rain_model.pkl")
temp_model = joblib.load("models/temp_model.pkl")

# ---------------- WEATHER API ----------------
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "current_temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "temp_min": data["main"]["temp_min"],
        "temp_max": data["main"]["temp_max"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
        "icon": data["weather"][0]["icon"],
        "main": data["weather"][0]["main"].lower(),
        "visibility": data.get("visibility", 10000) / 1000,
        "clouds": data.get("clouds", {}).get("all", 0)
    }

# ---------------- ROUTES ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    weather = None
    future = None
    error = None

    if request.method == 'POST':
        try:
            city = request.form['city']
            print(f"User entered city: {city}")
            
            weather = get_current_weather(city)
            print(f"Weather data received: {weather}")

            # Prepare input for rain model
            input_df = pd.DataFrame([{
                "MinTemp": weather["temp_min"],
                "MaxTemp": weather["temp_max"],
                "WindGustDir": 0,
                "WindGustSpeed": weather["wind_speed"],
                "Humidity": weather["humidity"],
                "Pressure": weather["pressure"],
                "Temp": weather["current_temp"]
            }])

            rain_pred = rain_model.predict(input_df)[0]
            weather['rain'] = 'Yes' if rain_pred else 'No'

            # Future temperature
            temps = []
            val = weather['current_temp']
            for _ in range(5):
                val = temp_model.predict([[val]])[0]
                temps.append(round(val, 1))

            tz = pytz.timezone("Asia/Kolkata")
            now = datetime.now(tz)
            times = [(now + timedelta(hours=i+1)).strftime("%H:00") for i in range(5)]
            future = list(zip(times, temps))
            print("Predictions completed successfully")
            
        except Exception as e:
            error = f"Error: {str(e)}"
            print(f"Error occurred: {error}")
            weather = None
            future = None

    return render_template('index.html', weather=weather, future=future, error=error)

if __name__ == '__main__':
    app.run(debug=True)
