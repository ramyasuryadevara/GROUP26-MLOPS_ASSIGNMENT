from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
# import mlflow.pyfunc  # Uncomment if using MLflow
import joblib
import logging
from datetime import datetime
import sqlite3
import os

# Ensure the log directory exists
os.makedirs("irislogs", exist_ok=True)

# Load MLflow model
# model_uri = "runs:/70968cdab4644053835a226c51eec164/model"
# model = mlflow.pyfunc.load_model(model_uri)

# OR load local model:
model = joblib.load("models/RandomForest.pkl")

# Setup file logging
logging.basicConfig(
    filename='irislogs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# SQLite DB setup
conn = sqlite3.connect("irislogs/predictions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS irislogs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    inputs TEXT,
    prediction TEXT
)
''')
conn.commit()

# FastAPI setup
app = FastAPI()

# Expected input features
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris prediction API is running."}

@app.post("/predict")
def predict(data: IrisRequest):
    # Create DataFrame
    input_dict = {
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width
    }
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)
    predicted_class = int(prediction[0])

    # Log to file
    log_msg = f"Input: {input_dict} | Prediction: {predicted_class}"
    logging.info(log_msg)

    # Log to SQLite
    cursor.execute('''
        INSERT INTO irislogs (timestamp, inputs, prediction)
        VALUES (?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        str(input_dict),
        str(predicted_class)
    ))
    conn.commit()

    return {"predicted_class": predicted_class}

@app.get("/metrics")
def metrics():
    cursor.execute("SELECT COUNT(*) FROM irislogs")
    total_requests = cursor.fetchone()[0]

    return {
        "total_predictions": total_requests,
        "last_updated": datetime.now().isoformat()
    }
