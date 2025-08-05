# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from AssingmentCode.predict import predict_transaction  # Import prediction function from predict.py
import logging
from datetime import datetime
import os
os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    filename="logs/predictions.log",   # File path for the logs
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# 🚀 Initialize FastAPI app
app = FastAPI()

# 📦 Define request body schema using Pydantic
class Transaction(BaseModel):
    transaction_amount: float
    transaction_time: int
    user_location: str
    merchant_location: str
    card_present: int  # 0 or 1
    device_type: str   # e.g., 'mobile', 'web', 'ATM'
    user_id: str       # e.g., 'user_123'

# 🔍 Prediction endpoint
@app.post("/predict")
def predict_endpoint(transaction: Transaction):
    input_data = transaction.dict()
    result = predict_transaction(input_data)

    # 📝 Log the request and result
    logging.info(f"INPUT: {input_data} | OUTPUT: {result}")
    
    return result

# ✅ Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "running"}
