import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"  # Adjust if your FastAPI runs elsewhere

st.title("Housing price prediction")

# Input fields for the iris features
total_rooms = st.number_input("total_rooms", min_value=0.0, format="%.2f")
total_bedrooms = st.number_input("total_bedrooms", min_value=0.0, format="%.2f")
population = st.number_input("population", min_value=0.0, format="%.2f")
households = st.number_input("households", min_value=0.0, format="%.2f")
median_income = st.number_input("median_income", min_value=0.0, format="%.2f")
housing_median_age = st.number_input("Phousing_median_age", min_value=0.0, format="%.2f")
latitude = st.number_input("latitude", min_value=0.0, format="%.2f")
longitude = st.number_input("longitude", min_value=0.0, format="%.2f")

if st.button("Predict"):
    # Prepare JSON payload
    payload = {
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "housing_median_age": housing_median_age,
        "latitude": latitude,
        "longitude": longitude
    }
    
    # Send POST request to FastAPI backend
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        prediction = response.json().get("result")
        st.success(f"Predicted Iris Species: {prediction}")
    else:
        st.error("Failed to get prediction from the backend")