import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import logging
from datetime import datetime
import sqlite3
import joblib  # If using local .pkl file
import os

os.makedirs("housinglogs", exist_ok=True)

app = FastAPI()

# model_uri = "runs:/4bf65a1a6fdd4d9fb80d35b460d5d721/model"
# model = mlflow.pyfunc.load_model(model_uri)

model = joblib.load("models/DecisionTree.pkl")

logging.basicConfig(
    filename='housinglogs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

conn = sqlite3.connect("housinglogs/predictions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS housinglogs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    inputs TEXT,
    prediction TEXT
)
''')
conn.commit()

prediction_count = 0

class HousingRequest(BaseModel):
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    housing_median_age: float
    latitude: float
    longitude: float

@app.get("/")
def root():
    return {"message": "Housing price prediction API is running."}

@app.post("/predict")
def predict(data: HousingRequest):
    global prediction_count
    prediction_count += 1

    df = pd.DataFrame([data.dict()])

    # Feature engineering
    df["AveRooms"] = df["total_rooms"] / df["households"]
    df["AveBedrms"] = df["total_bedrooms"] / df["households"]
    df["AveOccup"] = df["population"] / df["households"]

    # Rename columns
    df.rename(columns={
        "median_income": "MedInc",
        "housing_median_age": "HouseAge",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "population": "Population"
    }, inplace=True)

    final_features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]

    prediction = model.predict(df[final_features])[0]

    # Log to file and SQLite
    input_data = data.dict()
    logging.info(f"Input: {input_data} | Prediction: {prediction}")
    cursor.execute("INSERT INTO housinglogs (timestamp, inputs, prediction) VALUES (?, ?, ?)",
                   (datetime.utcnow().isoformat(), str(input_data), str(prediction)))
    conn.commit()

    return {"predicted_price": float(prediction)}

@app.get("/metrics")
def metrics():
    return {"total_predictions": prediction_count}
