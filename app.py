from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained model bundle
model_bundle = joblib.load("model.pkl")

# Extract models and scaler
wind_model = model_bundle["wind_model"]
intensity_model = model_bundle["intensity_model"]
scaler = model_bundle["scaler"]

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "🌪️ Tauktae Cyclone Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [
            data["Latitude"], data["Longitude"], data["Pressure"],
            data["Temperature"], data["Humidity"], data["Hour"],
            data["Day"], data["WindSpeed_lag"], data["Pressure_lag"],
            data["TempHumidityIndex"]
        ]

        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)

        wind_pred = wind_model.predict(input_scaled)[0]
        intensity_pred = int(intensity_model.predict(input_scaled)[0])

        categories = ["Tropical Depression", "Tropical Storm", "Category 1-2", "Category 3+"]
        risk = "HIGH" if wind_pred > 64 else "MODERATE" if wind_pred > 34 else "LOW"

        return jsonify({
            "wind_speed_knots": round(float(wind_pred), 2),
            "intensity_category": intensity_pred,
            "intensity_label": categories[intensity_pred],
            "risk_level": risk
        })

    except KeyError as ke:
        return jsonify({"error": f"Missing key in input JSON: {str(ke)}"}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)