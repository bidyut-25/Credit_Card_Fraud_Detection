from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model and the scaler
model = joblib.load('fraud_model_tuned.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request in JSON format
        data = request.json
        transaction_df = pd.DataFrame(data, index=[0])

        # Preprocess the data using the saved scaler
        transaction_df[['Time', 'Amount']] = scaler.transform(transaction_df[['Time', 'Amount']])
        
        # Make a prediction using the loaded model
        prediction = model.predict(transaction_df)
        
        # Prepare the response
        result = 'Fraudulent' if prediction[0] == 1 else 'Legitimate'
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)