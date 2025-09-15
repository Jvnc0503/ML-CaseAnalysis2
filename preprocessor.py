import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("used_cars.csv")

# region Preprocessing

# Replace prefix
df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '').astype(int)
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Feature extraction
df['hp'] = df['engine'].str.extract(r'(\d+\.\d+)HP').astype(float, errors='ignore')
df['engine_displacement'] = df['engine'].str.extract(r'(\d+\.\d+)\s*L')
df['engine_displacement'] = df['engine_displacement'].fillna(df['engine'].str.extract(r'(\d+\.\d+)\s*LITER')[0])
df['engine_displacement'] = df['engine_displacement'].astype(float, errors='ignore')
df['is_v_engine'] = df['engine'].str.contains(r'V\d+', case=False, na=False)

# Clean fuel_type
df['fuel_type'] = df['fuel_type'].str.strip().str.upper().replace({'PLUG-IN HYBRID': 'HYBRID', 'NOT SUPPORTED':'OTHER', '–':'OTHER'})

# Clean transmission
def classify_transmission(t: str) -> str:
    t = t.upper()
    if 'M/T' in t or 'MT' in t or 'MANUAL' in t:
        return 'MT'
    elif 'A/T' in t or 'AT' in t or 'AUTOMATIC' in t:
        return 'AT'
    elif 'CVT' in t or 'VARIABLE' in t or 'SINGLE-SPEED' in t:
        return 'CVT'
    else:
        return 'OTHER'
    
df['transmission'] = df['transmission'].apply(classify_transmission)

# region Handle missing values

# Assigns mean hp per brand, if brand has no hp values, drop those rows
df['hp'] = df.groupby('brand')['hp'].transform(lambda x: x.fillna(x.mean()))
df.dropna(subset=['hp'], inplace=True)

# For categorical features, fill missing with most common value per brand, if still missing, fill with 'OTHER'
most_common_fuel = df.groupby('brand')['fuel_type'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
df['fuel_type'] = df.apply(
    lambda row: most_common_fuel[row['brand']] if pd.isna(row['fuel_type']) else row['fuel_type'],
    axis=1
)
df['fuel_type'] = df['fuel_type'].fillna('OTHER')

# For numerical features, fill missing with most common value per brand, if still missing, fill with median
most_common_displacement = df.groupby('brand')['engine_displacement'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
df['engine_displacement'] = df.apply(
    lambda row: most_common_displacement[row['brand']] if pd.isna(row['engine_displacement']) else row['engine_displacement'],
    axis=1
)
df['engine_displacement'] = df['engine_displacement'].fillna(df['engine_displacement'].median())

# region Remove outliers
columns = ['engine_displacement', 'hp', 'price', 'milage']
for col in columns:
    Q1 = df[col].quantile(0.25)  
    Q3 = df[col].quantile(0.75)   
    IQR = Q3 - Q1                    
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# region Encode categorical features
df['accident'] = df['accident'].apply(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)
df['clean_title'] = df['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)
categorical_columns = ['fuel_type', 'transmission', 'is_v_engine']

for cat_col in categorical_columns:
    encoder = LabelEncoder()
    df[cat_col] = encoder.fit_transform(df[cat_col])

# region Feature engineering
df['age'] = 2025 - df['model_year']
#df['milage_per_year'] = df.apply(lambda row: row['milage'] / row['age'] if row['age'] > 0 else row['milage'], axis=1)
#df['age_bin'] = pd.qcut(df['age'], q=4, labels=['New', 'Mid', 'Old', 'Very Old'])
#df['milage_bin'] = pd.qcut(df['milage'], q=4, labels=['Low', 'Mid', 'High', 'Very High'])

# Drop features
df.drop(columns=['model', 'model_year', 'engine', 'ext_col', 'int_col'], axis=1, inplace=True)

df.to_csv('used_cards_clean.csv', index=False)