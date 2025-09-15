import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("used_cars.csv")

df = df.copy()
df["age"] = 2025 - df["model_year"]

df['milage'] = df['milage'].str.replace('mi.', '', regex=False)   # quitar "mi."
df['milage'] = df['milage'].str.replace(',', '', regex=False)     # quitar comas
df['milage'] = df['milage'].str.strip()                           # quitar espacios
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')       # convertir a número

df['price'] = df['price'].str.replace('$', '', regex=False)   # quitar símbolo $
df['price'] = df['price'].str.replace(',', '', regex=False)   # quitar comas
df['price'] = pd.to_numeric(df['price'], errors='coerce')     # convertir a número


# Separate features and target variable
y = df["price"]
x = df.drop("price", axis=1)

# Select numerical and categorical features
num_features = ["age", "milage"]
cat_features = ["brand", "model", "transmission", "fuel_type"]


# Preprocessing pipeline for numerical features
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing pipeline for categorical features
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Full pipeline with model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# Data split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")