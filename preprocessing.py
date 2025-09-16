# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(input_file="used_cars.csv", output_file="used_cars_processed.csv"):
    df = pd.read_csv(input_file)

    # --- Clean numeric fields ---
    df['milage'] = (
        df['milage']
        .str.replace(' mi.', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(int)
    )
    df['price'] = (
        df['price']
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    # --- Feature extraction ---
    df['hp'] = df['engine'].str.extract(r'(\d+\.\d+)HP').astype(float, errors='ignore')
    df['engine_displacement'] = df['engine'].str.extract(r'(\d+\.\d+)\s*L')
    df['engine_displacement'] = df['engine_displacement'].fillna(
        df['engine'].str.extract(r'(\d+\.\d+)\s*LITER')[0]
    )
    df['engine_displacement'] = df['engine_displacement'].astype(float, errors='ignore')
    df['is_v_engine'] = df['engine'].str.contains(r'V\d+', case=False, na=False)

    # --- Clean fuel_type ---
    df['fuel_type'] = (
        df['fuel_type']
        .str.strip()
        .str.upper()
        .replace({'PLUG-IN HYBRID': 'HYBRID', 'NOT SUPPORTED': 'OTHER', '–': 'OTHER'})
    )

    # --- Clean transmission ---
    def classify_transmission(t: str) -> str:
        t = str(t).upper()
        if 'M/T' in t or 'MT' in t or 'MANUAL' in t:
            return 'MT'
        elif 'A/T' in t or 'AT' in t or 'AUTOMATIC' in t:
            return 'AT'
        elif 'CVT' in t or 'VARIABLE' in t or 'SINGLE-SPEED' in t:
            return 'CVT'
        else:
            return 'OTHER'
    
    df['transmission'] = df['transmission'].apply(classify_transmission)

    # --- Handle missing values ---
    df['hp'] = df.groupby('brand')['hp'].transform(lambda x: x.fillna(x.mean()))
    df.dropna(subset=['hp'], inplace=True)

    most_common_fuel = df.groupby('brand')['fuel_type'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )
    df['fuel_type'] = df.apply(
        lambda row: most_common_fuel[row['brand']] if pd.isna(row['fuel_type']) else row['fuel_type'],
        axis=1
    )
    df['fuel_type'] = df['fuel_type'].fillna('OTHER')

    most_common_disp = df.groupby('brand')['engine_displacement'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )
    df['engine_displacement'] = df.apply(
        lambda row: most_common_disp[row['brand']] if pd.isna(row['engine_displacement']) else row['engine_displacement'],
        axis=1
    )
    df['engine_displacement'] = df['engine_displacement'].fillna(df['engine_displacement'].median())

    # --- Remove outliers ---
    for col in ['engine_displacement', 'hp', 'price', 'milage']:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # --- Encode categorical features ---
    df['accident'] = df['accident'].apply(
        lambda x: 1 if x == 'At least 1 accident or damage reported' else 0
    )
    df['clean_title'] = df['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Encoders dictionary to save for later
    encoders = {}

    for col in ['fuel_type', 'transmission', 'is_v_engine', 'brand']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # --- Feature engineering ---
    df['age'] = 2025 - df['model_year']
    df.rename(columns={'milage': 'mileage'}, inplace=True)

    # Drop unused columns
    df.drop(columns=['model', 'model_year', 'engine', 'ext_col', 'int_col'], inplace=True)

    # Save cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"✅ Preprocessing complete. Clean data saved to {output_file}")

    # Save encoders for later use in training/Streamlit
    joblib.dump(encoders, "encoders.pkl")
    print("✅ Encoders saved to encoders.pkl")

if __name__ == "__main__":
    preprocess_data()
