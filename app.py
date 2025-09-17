from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        transaction_df = pd.DataFrame(data, index=[0])
        transaction_df[['Time', 'Amount']] = scaler.transform(transaction_df[['Time', 'Amount']])
        prediction = model.predict(transaction_df)
        result = "Fraudulent" if int(prediction[0]) == 1 else "Legitimate"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
