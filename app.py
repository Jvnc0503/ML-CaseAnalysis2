import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and encoders
model = joblib.load("used_cars_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Used Car Dealership 🚗")

user_type = st.radio("Who are you?", ('Seller', 'Buyer'))

# ---------------- SELLER ----------------
if user_type == 'Seller':
    st.header("Sell your car at a fair price")
    st.write("Enter your car details to estimate a transparent market price.")

    # --- Inputs ---
    brand = st.text_input("Brand", "Toyota")
    model_name = st.text_input("Model (not used by model)", "Corolla")
    year = st.number_input("Model Year", min_value=1980, max_value=2025, value=2015, step=1)
    mileage = st.number_input("Mileage (mi)", min_value=0, value=50000, step=1000)
    hp = st.number_input("Horsepower (hp)", min_value=50, value=200, step=10)
    displacement = st.number_input("Engine Displacement (L)", min_value=0.5, value=2.0, step=0.1)
    transmission = st.selectbox("Transmission Type", ['MT', 'AT', 'CVT', 'OTHER'])
    fuel_type = st.selectbox("Fuel Type", ['GASOLINE', 'DIESEL', 'HYBRID', 'ELECTRIC', 'OTHER'])
    accident = st.selectbox("Accident History", ['No accidents', 'At least 1 accident or damage reported'])
    clean_title = st.selectbox("Clean Title", ['Yes', 'No'])
    is_v_engine = st.selectbox("V Engine?", ['Yes', 'No'])

    # --- Button ---
    if st.button("Estimate Price"):
        # Prepare input row
        input_data = pd.DataFrame([{
            "brand": brand,
            "mileage": mileage,
            "hp": hp,
            "engine_displacement": displacement,
            "transmission": transmission,
            "fuel_type": fuel_type,
            "accident": 1 if accident == 'At least 1 accident or damage reported' else 0,
            "clean_title": 1 if clean_title == 'Yes' else 0,
            "is_v_engine": 1 if is_v_engine == 'Yes' else 0,
            "age": 2025 - year
        }])

        # Apply encoders (same as training)
        for col in ['fuel_type', 'transmission', 'is_v_engine', 'brand']:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels
                if input_data.at[0, col] not in le.classes_:
                    input_data.at[0, col] = le.classes_[0]  # fallback
                input_data[col] = le.transform(input_data[col])

        # Ensure feature order matches training
        feature_order = pd.read_csv("used_cars_processed.csv").drop(columns=["price"]).columns.tolist()
        input_data = input_data[feature_order]

        # Predict (model trained on log(price))
        pred_log = model.predict(input_data)[0]
        estimated_price = np.expm1(pred_log)

        st.success(f"💰 Estimated Market Price: ${estimated_price:,.2f}")

