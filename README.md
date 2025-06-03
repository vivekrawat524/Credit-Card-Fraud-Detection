# Credit-Card-Fraud-Detection
# Credit Card Fraud Detection

This project detects if a credit card transaction is fraud or not using machine learning.

---

## How It Works

- We use data of past transactions.
- Train a model to learn patterns of fraud.
- The model predicts if a new transaction is fraud or legitimate.

---

## How to Use

1. Download the dataset `creditcard.csv` from Kaggle.
2. Run `python train_model.py` to train the model.
3. Run the web app with:  
   `streamlit run app.py`
4. Open the link in your browser and enter transaction details to check for fraud.

---

## Files

- `train_model.py` — trains the model
- `app.py` — the web app to predict fraud
- `fraud_detection_model.pkl` — saved model file
- `creditcard.csv` — dataset file

---

## Requirements

Install required packages with:

