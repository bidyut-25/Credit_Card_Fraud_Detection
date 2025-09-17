# 💳 Credit Card Fraud Detection: An End-to-End Machine Learning Project

This project demonstrates a complete machine learning workflow to detect fraudulent credit card transactions. From data preprocessing to model deployment, the goal was to build a robust and actionable fraud detection system.

## ✨ Key Features

- **Data Handling:** Efficiently managed a highly imbalanced dataset using **SMOTE (Synthetic Minority Over-sampling Technique)** to avoid model bias.
- **Model Optimization:** Utilized **Hyperparameter Tuning with GridSearchCV** to find the optimal parameters for a **Random Forest Classifier**, significantly improving the model's performance.
- **Actionable Evaluation:** Evaluated the model using industry-relevant metrics like **Precision, Recall, and F1-score**, focusing on the model's ability to minimize **false negatives (missed fraud)**.
- **REST API:** Built a **RESTful API using Flask** to expose the model for real-time predictions.
- **Deployment:** Deployed the model to a live, publicly accessible endpoint to showcase a complete machine learning solution.

## 🛠️ Technologies Used

- Python
- scikit-learn
- imbalanced-learn
- Pandas & NumPy
- Flask
- joblib

## 📈 Results

The final model achieved strong performance on the unseen test data, with a focus on **high recall** for the fraudulent class to minimize missed fraud cases.

| Class       | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Legitimate | 1.00      | 1.00   | 1.00     | 5943    |
| Fraud      | 0.95      | 0.85   | 0.90     | 17      |

## 🚀 Live Demo

You can interact with the deployed model directly. Send a POST request with transaction data to the API endpoint below to get a real-time prediction.

**API Endpoint:** `https://your-ngrok-url.ngrok.io/predict`

### 🔧 Sample Python Script to Test the API

```python
import requests
import json

api_url = "https://your-ngrok-url.ngrok.io/predict"

test_transaction = {
    "Time": 400.0, "V1": -0.421, "V2": 0.123, "V3": 1.123, "V4": -0.456, "V5": -0.789,
    "V6": 0.123, "V7": -0.456, "V8": 0.789, "V9": -0.123, "V10": 0.456,
    "V11": -0.789, "V12": 0.123, "V13": -0.456, "V14": 0.789, "V15": -0.123,
    "V16": 0.456, "V17": -0.789, "V18": 0.123, "V19": -0.456, "V20": 0.789,
    "V21": -0.123, "V22": 0.456, "V23": -0.789, "V24": 0.123, "V25": -0.456,
    "V26": 0.789, "V27": -0.123, "V28": 0.456, "Amount": 150.0
}

response = requests.post(api_url, json=test_transaction)
print(response.json())
```

---

## 📂 Project Structure

```
.
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
├── model/
│   └── rf_model.joblib
|   └── scalar.pkl
├── app.py
├── requirements.txt
└── README.md
```

---
👨‍💻 Developed by [Bidyut Supakar]  
🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/your-profile)
