# 📦 Import Libraries
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

import joblib
import os

# 🎲 Reproducibility
np.random.seed(42)
random.seed(42)

# 📊 Generate Synthetic Transaction Data
n_samples = 10000
fraud_ratio = 0.03
n_frauds = int(n_samples * fraud_ratio)

# Transaction amounts (log-normal distribution clipped between 1000-100000)
transaction_amounts = np.round(np.random.lognormal(mean=10, sigma=1, size=n_samples), 2)
transaction_amounts = np.clip(transaction_amounts, 1000, 100000).astype(int)

# Transaction times (0 to 23 hours)
transaction_times = np.random.randint(0, 24, size=n_samples)

# Random user and merchant locations (Indian states)
indian_states = ['Gujarat', 'Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'Rajasthan', 'West Bengal']
user_locations = np.random.choice(indian_states, size=n_samples)
merchant_locations = np.random.choice(indian_states, size=n_samples)

# Card present flag (0 or 1)
card_present = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

# Device types
device_types = np.random.choice(['mobile', 'web', 'ATM'], size=n_samples, p=[0.5, 0.4, 0.1])

# User IDs
user_ids = [f'user_{np.random.randint(1, 501)}' for _ in range(n_samples)]

# Fraud labels (3% frauds)
is_fraud = np.array([1]*n_frauds + [0]*(n_samples - n_frauds))
np.random.shuffle(is_fraud)

# 🧾 Create DataFrame
data = pd.DataFrame({
    'transaction_amount': transaction_amounts,
    'transaction_time': transaction_times,
    'user_location': user_locations,
    'merchant_location': merchant_locations,
    'card_present': card_present,
    'device_type': device_types,
    'user_id': user_ids,
    'is_fraud': is_fraud
})

# 🔁 Feature Engineering
df = data.copy()

# Same state indicator
df['is_same_state'] = (df['user_location'] == df['merchant_location']).astype(int)

# User location risk score based on average fraud rate
location_fraud_rate = df.groupby('user_location')['is_fraud'].mean().to_dict()
df['location_risk_score'] = df['user_location'].map(location_fraud_rate)

# User behavior features
df['user_avg_amount'] = df.groupby('user_id')['transaction_amount'].transform('mean')
df['user_txn_count'] = df['user_id'].map(df['user_id'].value_counts())
df['device_freq_per_user'] = df.groupby(['user_id', 'device_type'])['device_type'].transform('count')
df['online_txn_ratio'] = df.groupby('user_id')['card_present'].transform(lambda x: 1 - x.mean())

# 🔠 Label Encoding for categorical variables
categorical_cols = ['user_location', 'merchant_location', 'device_type', 'user_id']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ⚖️ Feature Scaling
scaler = StandardScaler()
df[['transaction_amount', 'transaction_time', 'user_avg_amount']] = scaler.fit_transform(
    df[['transaction_amount', 'transaction_time', 'user_avg_amount']]
)

# 📊 Data Visualizations
data['fraud'] = data['is_fraud']

plt.figure(figsize=(10, 5))
sns.histplot(data=data, x='transaction_amount', bins=50, hue='fraud', multiple='stack')
plt.title("Transaction Amount Distribution by Fraud")
plt.xlabel("Amount (INR)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='transaction_time', hue='fraud')
plt.title("Fraud by Hour of Day")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Transaction Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='user_location', hue='fraud')
plt.title("Fraud Count by User Location")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='device_type', hue='fraud')
plt.title("Fraud by Device Type")
plt.show()

# 📈 Check class imbalance
data['is_fraud'].value_counts(normalize=True)

# 🎯 Feature & Target Split
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# ✂️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔄 Handle Imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 🌳 Random Forest Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)

# 🧪 Prediction & Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# ⚡ XGBoost Model (basic)
xgb_model = XGBClassifier(n_estimators=100, scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_bal, y_train_bal)

y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_prob > 0.2).astype(int)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
print("\nClassification Report:\n", classification_report(y_test, y_pred_custom))
print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# ⚡ XGBoost Model (tuned)
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    subsample=0.9,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

xgb_model.fit(X_train_bal, y_train_bal)

y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb > 0.2).astype(int)

print("🔍 XGBoost Evaluation")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))

# 🌲 Isolation Forest (Unsupervised)
iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
iso.fit(X_train[y_train == 0])

iso_preds = iso.predict(X_test)
y_pred_iso = np.where(iso_preds == -1, 1, 0)

print("🔍 Isolation Forest Evaluation")
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))
print("ROC AUC:", roc_auc_score(y_test, y_pred_iso))

# 🤝 Combine XGBoost + Isolation Forest
final_preds = ((y_pred_xgb == 1) | (y_pred_iso == 1)).astype(int)

print("🔍 Combined Model (XGB + ISO) Evaluation")
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds))
print("ROC AUC (XGB only):", roc_auc_score(y_test, y_prob_xgb))

# 💾 Save models & transformers
os.makedirs("model", exist_ok=True)

joblib.dump(xgb_model, "model/final_xgb_model.pkl")
joblib.dump(iso, "model/final_iso_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("All models and tools saved successfully.")
