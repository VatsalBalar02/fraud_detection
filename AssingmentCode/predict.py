# predict.py

import joblib
import numpy as np
import pandas as pd

# 🧠 Load trained models and preprocessing tools
xgb_model = joblib.load("model/final_xgb_model.pkl")
iso_model = joblib.load("model/final_iso_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/label_encoders.pkl")


# 📦 Preprocessing Function
def preprocess_input(data_dict):
    """
    Convert raw input dictionary to preprocessed DataFrame ready for prediction.
    """

    # Convert input dict to single-row DataFrame
    df = pd.DataFrame([data_dict])

    # Feature Engineering (approximate values for real-time prediction)
    df['is_same_state'] = (df['user_location'] == df['merchant_location']).astype(int)

    df['location_risk_score'] = 0.03  # assumed average risk score
    df['user_avg_amount'] = df['transaction_amount']  # default to current txn amount
    df['user_txn_count'] = 10  # assume average txn count
    df['device_freq_per_user'] = 5  # assume common usage count
    df['online_txn_ratio'] = 1 - df['card_present']  # card not present = online txn

    # Apply Label Encoding to categorical features
    for col in ['user_location', 'merchant_location', 'device_type', 'user_id']:
        df[col] = encoders[col].transform(df[col])

    # Scale numerical features
    df[['transaction_amount', 'transaction_time', 'user_avg_amount']] = scaler.transform(
        df[['transaction_amount', 'transaction_time', 'user_avg_amount']]
    )

    return df


# 🔮 Prediction Function
def predict_transaction(data_dict):
    """
    Predict whether a transaction is fraudulent using XGBoost and Isolation Forest.
    Returns both class prediction and fraud probability.
    """
    # Preprocess input
    df = preprocess_input(data_dict)

    # XGBoost prediction
    xgb_prob = xgb_model.predict_proba(df)[0][1]
    xgb_pred = int(xgb_prob > 0.2)  # threshold adjusted

    # Isolation Forest prediction
    iso_pred = 1 if iso_model.predict(df)[0] == -1 else 0  # -1 = anomaly/fraud

    # Combine both predictions (ensemble logic)
    final_pred = 1 if (xgb_pred or iso_pred) else 0

    return {
        "prediction": "fraud" if final_pred else "not fraud",
        "probability": float(round(xgb_prob, 2))  # probability from XGBoost
    }
