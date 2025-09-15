import streamlit as st
import pandas as pd
import numpy as np
import joblib

#model = joblib.load('')

st.title("Used Car Dealership 🚗")

user_type = st.radio("Who are you?", ('Seller', 'Buyer'))

# region Seller
if user_type == 'Seller':
    st.header("Sell your car at a fair price")
    st.write("Enter your car details to estimate a transparent market price.")

    brand = st.text_input("Brand", "Toyota" )
    model_name = st.text_input("Model", "Corolla")
    year = st.number_input("Model Year", min_value=1980, max_value=2025, value=2015, step=1)
    mileage = st.number_input("Mileage (mi)", min_value=0, value=50000, step=1000)
    hp = st.number_input("Horsepower (hp)", min_value=50, value=200, step=10)
    displacement = st.number_input("Engine Displacement (cc)", min_value=500, value=2000, step=100)
    transmission = st.selectbox("Transmission Type", ['Automatic', 'Manual', 'CVT', 'Other'])
    fuel_type = st.selectbox("Fuel Type", ['Gasoline', 'Diesel', 'Hybrid', 'Electric', 'Other'])
    accident = st.selectbox("Accident History", ['No accidents', 'At least 1 accident or damage reported'])
    clean_title = st.selectbox("Clean Title", ['Yes', 'No'])

    if st.button("Estimate Price"):
        st.write("Estimating price...")
        # Here you would typically call your model to predict the price
        # For demonstration, we'll use a dummy value
        estimated_price = 15000  # Replace with model prediction
        st.success(f"Estimated Market Price: ${estimated_price:,.2f}")