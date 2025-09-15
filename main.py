import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42

# -------- Utilidades de limpieza --------
def normalize_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False})
    )

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Columnas esperadas del dataset (según Kaggle): 
    # brand, model, model_year, milage, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title, price
    # Forzar tipos básicos
    for col in ["model_year", "milage", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalizar booleanos si vienen como texto/num
    for bcol in ["accident", "clean_title"]:
        if bcol in df.columns:
            df[bcol] = normalize_bool_series(df[bcol]).fillna(False)

    # Quitar filas sin target o numéricas claves
    df = df.dropna(subset=["price", "model_year", "milage"])
    return df

def eval_metrics(model, Xtr, Xts, ytr, yts):
    y_pred_tr = model.predict(Xtr)
    y_pred_ts = model.predict(Xts)
    rmse_tr = float(np.sqrt(mean_squared_error(ytr, y_pred_tr)))
    rmse_ts = float(np.sqrt(mean_squared_error(yts, y_pred_ts)))
    r2_tr = float(r2_score(ytr, y_pred_tr))
    r2_ts = float(r2_score(yts, y_pred_ts))
    return {
        "RMSE_train": rmse_tr, "RMSE_test": rmse_ts,
        "R2_train": r2_tr, "R2_test": r2_ts
    }

def main(args):
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(str(data_path))

    TARGET = "price"
    FEATURES = [c for c in df.columns if c != TARGET]

    X = df[FEATURES].copy()
    y = df[TARGET].astype(float)

    # Definir numéricas y categóricas
    num_cols = [c for c in X.columns if c in ["model_year", "milage"]]
    # Todas las demás, tratarlas como categóricas (incluye brand, model, engine, colors, booleans, etc.)
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    # -------- Pipelines de preprocesamiento + modelos --------
    # 1) Lineal: escalar numéricas, OHE categóricas
    preproc_linear = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe_linear = Pipeline([
        ("prep", preproc_linear),
        ("model", LinearRegression())
    ])

    # 2) Polinómica (grado 2) sobre numéricas + OHE categóricas
    preproc_poly = ColumnTransformer(
        transformers=[
            ("num_poly", Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe_poly = Pipeline([
        ("prep", preproc_poly),
        ("model", LinearRegression())
    ])

    # 3) Ridge (CV) con escaleo + OHE
    preproc_ridge = preproc_linear  # mismo preproc que lineal
    ridge_alphas = np.logspace(-3, 3, 13)
    pipe_ridge = Pipeline([
        ("prep", preproc_ridge),
        ("model", RidgeCV(alphas=ridge_alphas, store_cv_values=False))
    ])

    # 4) Lasso (CV) con escaleo + OHE
    preproc_lasso = preproc_linear
    lasso_alphas = np.logspace(-3, 1, 9)  # Lasso necesita más cuidado con alpha
    pipe_lasso = Pipeline([
        ("prep", preproc_lasso),
        ("model", LassoCV(alphas=lasso_alphas, max_iter=20000, random_state=RANDOM_STATE))
    ])

    candidates = {
        "Linear": pipe_linear,
        "PolyDeg2": pipe_poly,
        "RidgeCV": pipe_ridge,
        "LassoCV": pipe_lasso,
    }

    report_rows = []
    fitted = {}

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        metrics = eval_metrics(pipe, X_train, X_test, y_train, y_test)

        extra = {}
        if name == "RidgeCV":
            extra["alpha_"] = float(pipe.named_steps["model"].alpha_)
        if name == "LassoCV":
            extra["alpha_"] = float(pipe.named_steps["model"].alpha_)

        row = {"model": name, **metrics, **extra}
        report_rows.append(row)
        fitted[name] = pipe

    report = pd.DataFrame(report_rows).sort_values("RMSE_test").reset_index(drop=True)
    print("\n=== Comparación de modelos (ordenado por RMSE_test) ===")
    print(report.to_string(index=False))

    # Elegir mejor por RMSE_test
    best_name = report.iloc[0]["model"]
    best_pipe = fitted[best_name]

    # Guardar artefactos
    joblib.dump(
        {"pipeline": best_pipe, "num_cols": num_cols, "cat_cols": cat_cols, "features": FEATURES},
        outdir / "best_model.pkl"
    )
    report.to_csv(outdir / "model_report.csv", index=False)

    summary = {
        "best_model": best_name,
        "metrics": report.iloc[0].to_dict(),
        "all_models": report_rows
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Mejor modelo: {best_name}")
    print(f"   Artefactos guardados en: {outdir.resolve()}/")
    print("   - best_model.pkl (pipeline completo)")
    print("   - metrics.json")
    print("   - model_report.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos de regresión (Lineal/Poli/Ridge/Lasso) y escoger el mejor.")
    parser.add_argument("--data", type=str, default="used_cars.csv", help="Ruta al CSV de autos (por defecto: used_cars.csv)")
    parser.add_argument("--outdir", type=str, default="artifacts", help="Carpeta de salida de artefactos (por defecto: artifacts/)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción de test (por defecto: 0.2)")
    args = parser.parse_args()
    main(args)
