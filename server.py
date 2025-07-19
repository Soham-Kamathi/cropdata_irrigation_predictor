# server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("crop_irrigation_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([
        data['MOI'],
        data['temp'],
        data['humidity'],
        data['soil_type_encoded'],
        data['stage_encoded'],
        data['crop_id_encoded']
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    return jsonify({"irrigate": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
