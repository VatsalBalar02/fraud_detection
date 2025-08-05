# 🛡️ Fraudulent Transaction Detection API

A full-stack machine learning API built with FastAPI to detect fraudulent credit card transactions using a hybrid model (XGBoost + Isolation Forest) trained on synthetic data.

---

## 📌 Features

- Synthetic dataset generation with behavioral and location-based fraud features
- Class imbalance handling using SMOTE
- Ensemble prediction with XGBoost (probability-based) + Isolation Forest (anomaly detection)
- REST API built using FastAPI
- `/predict` endpoint returns fraud prediction and confidence score
- Designed for modularity, logging, and reusability

---

## 📁 Project Structure

```
AssingmentOfOmTech/
├── AssingmentCode/
│   ├── predict.py
│   ├── train.py
├── model/
│   ├── final_xgb_model.pkl
│   ├── final_iso_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── logs/
│   ├── predictions.log
├── main.py              # FastAPI app
├── predict.py           # Model inference logic
├── requirements.txt     # Dependencies
└── README.md            # You're here
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository or unzip the folder:
```bash
git clone https://github.com/vatsalbalar02/AssingmentOfOmTech.git
cd fraud-api
```

### 2️⃣ Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### 4️⃣ Start the API server:
```bash
uvicorn main:app --reload
```

---

## 🔍 API Documentation (Swagger UI)

Once the server is running, visit:

📄 **Swagger UI**: http://127.0.0.1:8000/docs  
🔁 **Health Check**: http://127.0.0.1:8000/health

---

## 🔁 Example Request

**POST** `/predict`

### ✅ Sample Input:
```json
{
  "transaction_amount": 14200,
  "transaction_time": 14,
  "user_location": "Gujarat",
  "merchant_location": "Delhi",
  "card_present": 0,
  "device_type": "mobile",
  "user_id": "user_42"
}
```

### ✅ Sample Response:
```json
{
  "prediction": "fraud",
  "probability": 0.87
}
```

---

## 🧠 Model Overview

- **XGBoost**: Handles imbalanced classification with `scale_pos_weight`, outputs fraud probability
- **Isolation Forest**: Detects anomalous patterns in transaction behavior
- **Final decision**: If either model detects fraud, label it as fraud (`OR` logic)

---

## ✅ Training Overview

The model was trained on a synthetic dataset of 10,000 transactions with the following features:

- `transaction_amount`, `transaction_time`, `user_location`, `merchant_location`
- `card_present`, `device_type`, `user_id`
- Engineered features: `is_same_state`, `location_risk_score`, `user_avg_amount`, `user_txn_count`, `device_freq_per_user`, `online_txn_ratio`

Class imbalance was handled using SMOTE + threshold tuning.

---

## 🔒 Security Notice

This is a prototype. In a production system:
- Secure endpoint with auth
- Sanitize inputs
- Use real-time logging & monitoring
- Replace synthetic data with real transactions

---

## 🧑‍💻 Author

**Developer:** Vatsal Balar 
**Role:** AI/ML Developer  
**Assignment:** OmTech Practical Task – Fraud Detection ML API  
**Stack:** Python, XGBoost, FastAPI, Scikit-learn, Pandas, Uvicorn

---

## 📃 License

This project is released for academic and demo purposes.
