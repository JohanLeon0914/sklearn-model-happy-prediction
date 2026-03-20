import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = joblib.load("./out/best_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    columns = [
        "high", "low", "gdp", "family", "lifexp",
        "freedom", "generosity", "corruption", "dystopia"
    ]
    
    features = pd.DataFrame([data["features"]], columns=columns)
    
    prediction = model.predict(features)
    
    return jsonify({"prediction": prediction.tolist()})
