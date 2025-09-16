import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model + bias and encoders
data = joblib.load("used_cars_model.pkl")
model = data["model"]
bias = data["bias"]

encoders = joblib.load("encoders.pkl")

st.title("Used Car Dealership 🚗")

user_type = st.radio("Who are you?", ('Seller', 'Buyer'))

# ---------------- SELLER ----------------
if user_type == 'Seller':
    st.header("Sell your car at a fair price")
    st.write("Enter your car details to estimate a transparent market price.")

    brand = st.text_input("Brand", "Toyota")
    model_name = st.text_input("Model (not used by model)", "Corolla")
    year = st.number_input("Model Year", min_value=1980, max_value=2025, value=2015, step=1)
    mileage = st.number_input("Mileage (mi)", min_value=0, value=50000, step=1000)
    hp = st.number_input("Horsepower (hp)", min_value=50, value=200, step=10)
    displacement = st.number_input("Engine Displacement (L)", min_value=1.0, value=2.0, step=0.1)
    transmission = st.selectbox("Transmission Type", ['MT', 'AT', 'CVT', 'OTHER'])
    fuel_type = st.selectbox("Fuel Type", ['GASOLINE', 'DIESEL', 'HYBRID', 'ELECTRIC', 'OTHER'])
    accident = st.selectbox("Accident History", ['No accidents', 'At least 1 accident or damage reported'])
    clean_title = st.selectbox("Clean Title", ['Yes', 'No'])
    is_v_engine = st.selectbox("V Engine?", ['Yes', 'No'])

    if st.button("Estimate Price"):
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

        # Apply encoders
        for col in ['fuel_type', 'transmission', 'is_v_engine', 'brand']:
            if col in encoders:
                le = encoders[col]
                if input_data.at[0, col] not in le.classes_:
                    input_data.at[0, col] = le.classes_[0]
                input_data[col] = le.transform(input_data[col])

        # Match training feature order
        feature_order = pd.read_csv("used_cars_processed.csv").drop(columns=["price"]).columns.tolist()
        input_data = input_data[feature_order]

        # Predict and correct with bias
        pred_log = model.predict(input_data)[0]
        estimated_price = np.expm1(pred_log) - bias

        st.success(f"💰 Estimated Market Price: ${estimated_price:,.0f}")
        st.caption("Based on historical sales of similar cars (bias corrected).")

# ---------------- BUYER ----------------
elif user_type == 'Buyer':
    st.header("Find the Best Car Deals")
    st.write("Upload a CSV of car listings or use our default dataset.")

    uploaded_file = st.file_uploader("Upload your car listings CSV", type=["csv"])

    if uploaded_file is not None:
        listings = pd.read_csv(uploaded_file)
    else:
        st.info("No file uploaded. Using default sample dataset `buyer_listings.csv`.")
        listings = pd.read_csv("buyer_listings.csv")

    original_listings = listings.copy()

    # Apply encoders
    for col in ['fuel_type', 'transmission', 'is_v_engine', 'brand']:
        if col in listings.columns and col in encoders:
            le = encoders[col]
            listings[col] = listings[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            listings[col] = le.transform(listings[col])

    # Match training feature order
    feature_order = pd.read_csv("used_cars_processed.csv").drop(columns=["price"]).columns.tolist()
    X = listings[feature_order]

    # Predictions with bias correction
    pred_log = model.predict(X)
    fair_prices = np.expm1(pred_log) - bias

    original_listings["Predicted Price"] = fair_prices.round(0).astype(int)
    original_listings["Deal"] = np.where(
        original_listings["price"] < original_listings["Predicted Price"] * 0.9,
        "✅ Good Deal",
        np.where(
            original_listings["price"] > original_listings["Predicted Price"] * 1.1,
            "❌ Overpriced",
            "⚖️ Fair Price"
        )
    )

    st.subheader("Recommended Listings")
    st.dataframe(original_listings)

    st.subheader("Top Good Deals")
    good_deals = original_listings[original_listings["Deal"] == "✅ Good Deal"]
    if not good_deals.empty:
        st.dataframe(good_deals.sort_values("Predicted Price"))
    else:
        st.info("No significantly undervalued cars found in this list.")
