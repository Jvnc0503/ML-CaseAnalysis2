# training.py
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.base import clone

import xgboost as xgb

RANDOM_STATE = 42

# -------------------- Utilidades de limpieza --------------------
def normalize_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .map({"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False})
    )

def clean_numeric_like(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r'^\s*([0-9]+)\s*k\s*$', lambda m: str(int(m.group(1)) * 1000), regex=True)
    s = (s
         .str.replace(r'[\$,€£]', '', regex=True)
         .str.replace(r'(mi|miles|km|kms|kilometros|kilómetros)', '', regex=True)
         .str.replace(r'[^\d\.-]', '', regex=True)
         .str.replace(r'\s+', '', regex=True))
    return pd.to_numeric(s, errors='coerce')

def parse_engine_liters(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    cc = s.str.extract(r'(\d{3,4})\s*cc', expand=False)
    liters_from_cc = pd.to_numeric(cc, errors='coerce') / 1000.0
    l_pat = s.str.extract(r'(\d+(?:[\.,]\d+)?)\s*l', expand=False)
    liters_from_l = (
        l_pat.str.replace(',', '.', regex=False).astype(float)
        .where(~l_pat.isna(), other=np.nan)
    )
    out = liters_from_l
    out = out.where(~out.isna(), liters_from_cc)
    return out

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    col_map_candidates = {
        "price":        ["price", "precio", "listing_price", "price_usd", "precio_usd"],
        "model_year":   ["model_year", "year", "modelyear", "año", "ano", "model year"],
        "mileage":      ["mileage", "milage", "miles", "kilometraje", "km", "odometer"],
        "accident":     ["accident", "accident_history", "has_accident", "accidents"],
        "clean_title":  ["clean_title", "clean title", "title_status", "clean-title", "title"],
        "engine":       ["engine", "motor", "engine_size"],
    }
    def pick_first_present(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_price      = pick_first_present(col_map_candidates["price"])
    col_model_year = pick_first_present(col_map_candidates["model_year"])
    col_mileage    = pick_first_present(col_map_candidates["mileage"])
    col_accident   = pick_first_present(col_map_candidates["accident"])
    col_clean      = pick_first_present(col_map_candidates["clean_title"])
    col_engine     = pick_first_present(col_map_candidates["engine"])

    missing_core = [name for name, col in
                    [("price", col_price), ("model_year", col_model_year), ("mileage", col_mileage)]
                    if col is None]
    if missing_core:
        raise ValueError(f"Faltan columnas clave en el CSV: {missing_core}. Columnas encontradas: {list(df.columns)}")

    rename_map = {col_price: "price", col_model_year: "model_year", col_mileage: "mileage"}
    if col_accident: rename_map[col_accident] = "accident"
    if col_clean:    rename_map[col_clean] = "clean_title"
    if col_engine:   rename_map[col_engine] = "engine"
    df = df.rename(columns=rename_map)

    df["price"] = clean_numeric_like(df["price"])
    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    df["mileage"] = clean_numeric_like(df["mileage"])

    if "accident" in df.columns:
        df["accident"] = normalize_bool_series(df["accident"]).fillna(False)
    if "clean_title" in df.columns:
        df["clean_title"] = normalize_bool_series(df["clean_title"]).fillna(False)

    current_year = datetime.now().year
    df.loc[~df["model_year"].between(1980, current_year + 1), "model_year"] = np.nan
    df.loc[(df["mileage"] < 0) | (df["mileage"] > 1_000_000), "mileage"] = np.nan
    df.loc[(df["price"] < 100) | (df["price"] > 500_000), "price"] = np.nan

    df["age"] = (current_year - df["model_year"]).clip(lower=0)
    df["mileage_per_year"] = df["mileage"] / df["age"].replace(0, 1)

    if "engine" in df.columns:
        df["engine_liters"] = parse_engine_liters(df["engine"])
    else:
        df["engine_liters"] = np.nan

    before = len(df)
    df = df.dropna(subset=["price", "model_year", "mileage"])
    after = len(df)
    print(f"[load_and_clean] Filas antes: {before} | después: {after}")
    if after == 0:
        raise ValueError("Después de limpiar, no quedan filas. Revisa valores nulos o el mapeo de columnas.")
    return df

def eval_metrics(y_true_tr, y_pred_tr, y_true_ts, y_pred_ts):
    rmse_tr = float(np.sqrt(mean_squared_error(y_true_tr, y_pred_tr)))
    rmse_ts = float(np.sqrt(mean_squared_error(y_true_ts, y_pred_ts)))
    mae_tr  = float(mean_absolute_error(y_true_tr, y_pred_tr))
    mae_ts  = float(mean_absolute_error(y_true_ts, y_pred_ts))
    r2_tr   = float(r2_score(y_true_tr, y_pred_tr))
    r2_ts   = float(r2_score(y_true_ts, y_pred_ts))
    return {"RMSE_train": rmse_tr, "RMSE_test": rmse_ts,
            "MAE_train": mae_tr,   "MAE_test": mae_ts,
            "R2_train": r2_tr,     "R2_test": r2_ts}

# -------------------- Wrapper: Booster dentro de Pipeline --------------------
class TrainedXGBRegressor:
    def __init__(self, booster, best_iteration=None):
        self.booster = booster
        self.best_iteration = best_iteration
    def predict(self, X):
        it_range = (0, int(self.best_iteration) + 1) if self.best_iteration is not None else None
        return self.booster.inplace_predict(X, iteration_range=it_range)

# --- constraints monótonos sobre columnas numéricas transformadas ---
def build_monotone_constraints(preproc_fitted, num_cols):
    names = preproc_fitted.get_feature_names_out()
    constraint = np.zeros(len(names), dtype=int)
    intended = {"model_year": +1, "age": -1, "mileage": -1, "mileage_per_year": -1, "engine_liters": +1}
    for i, n in enumerate(names):
        if n.startswith("num__"):
            raw = n.replace("num__", "")
            if raw in intended:
                constraint[i] = intended[raw]
    return "(" + ",".join(str(int(x)) for x in constraint) + ")"

def main(args):
    ROOT = Path(__file__).resolve().parent        # ← raíz del repo/proyecto
    outdir = Path(args.outdir)                    # ← artifacts/ u otro
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar y limpiar
    df = load_and_clean(str(args.data))

    # === Guardar dataset limpio en la RAÍZ ===
    clean_csv_root = ROOT / "dataset_clean.csv"
    df.to_csv(clean_csv_root, index=False, encoding="utf-8")
    print(f"[save_clean] CSV limpio (root): {clean_csv_root.resolve()}")

    # (Opcional) copia en artifacts para compatibilidad
    try:
        df.to_csv(outdir / "dataset_clean.csv", index=False, encoding="utf-8")
    except Exception as e:
        print(f"[warn] no se pudo guardar copia en artifacts: {e}")

    # 2) Preparar datos
    TARGET = "price"
    FEATURES = [c for c in df.columns if c != TARGET]
    X = df[FEATURES].copy()
    y = df[TARGET].astype(float)

    base_num = ["model_year", "mileage", "age", "mileage_per_year", "engine_liters"]
    num_cols = [c for c in X.columns if c in base_num]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Splits
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )
    y_train_full_log = np.log1p(y_train_full)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full_log, test_size=0.2, random_state=RANDOM_STATE
    )

    # Preprocesamiento
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    try:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)),
        ])
    except TypeError:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)

    preproc_fitted = clone(preproc).fit(X_train, y_train)
    X_train_tr       = preproc_fitted.transform(X_train)
    X_valid_tr       = preproc_fitted.transform(X_valid)
    X_test_tr        = preproc_fitted.transform(X_test)
    X_train_full_tr  = preproc_fitted.transform(X_train_full)

    # Modelo
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": 0.02,
        "max_depth": 5,
        "min_child_weight": 8,
        "subsample": 0.7890847779356666,
        "colsample_bytree": 0.7085437831660171,
        "lambda": 1.350599677264404,
        "alpha": 0.11800590212051532,
        "monotone_constraints": build_monotone_constraints(preproc_fitted, num_cols),
        "seed": RANDOM_STATE,
        "verbosity": 0,
    }

    num_boost_round = 10000
    early_stop = xgb.callback.EarlyStopping(rounds=200, metric_name="rmse", data_name="Valid", save_best=True)

    D_train = xgb.DMatrix(X_train_tr, label=y_train.values)
    D_valid = xgb.DMatrix(X_valid_tr, label=y_valid.values)
    D_train_full = xgb.DMatrix(X_train_full_tr)
    D_test = xgb.DMatrix(X_test_tr)

    booster = xgb.train(
        params=params,
        dtrain=D_train,
        num_boost_round=num_boost_round,
        evals=[(D_train, "Train"), (D_valid, "Valid")],
        callbacks=[early_stop],
        verbose_eval=False,
    )

    best_iteration = getattr(booster, "best_iteration", None)
    it_range = (0, int(best_iteration) + 1) if best_iteration is not None else None

    # 3) Métricas en escala real (des-log)
    y_pred_train_full_log = booster.inplace_predict(X_train_full_tr, iteration_range=it_range)
    y_pred_test_log       = booster.inplace_predict(X_test_tr,       iteration_range=it_range)
    y_pred_train_full = np.expm1(y_pred_train_full_log)
    y_pred_test       = np.expm1(y_pred_test_log)

    metrics = eval_metrics(y_train_full.values, y_pred_train_full, y_test.values, y_pred_test)
    report_rows = [{
        "model": "XGBoost(train API, bestparams+features)",
        **metrics,
        "best_iteration_": int(best_iteration) if best_iteration is not None else -1,
        "best_score_valid_rmse": float(getattr(booster, "best_score", np.nan)),
    }]
    report = pd.DataFrame(report_rows).sort_values("RMSE_test").reset_index(drop=True)
    print("\n=== Comparación de modelos (ordenado por RMSE_test) ===")
    print(report.to_string(index=False))

    # 4) Artefactos
    try:
        booster.save_model(str(outdir / "booster.json"))
    except Exception as e:
        print(f"[warn] no se pudo guardar booster.json: {e}")

    model_wrapper = TrainedXGBRegressor(booster=booster, best_iteration=best_iteration)
    best_pipe = Pipeline([("prep", preproc_fitted), ("model", model_wrapper)])

    # === Guardar en la RAÍZ: SOLO .pkl y CSV limpio ===
    best_model_root = ROOT / "best_model.pkl"
    joblib.dump(
        {"pipeline": best_pipe, "num_cols": num_cols, "cat_cols": cat_cols, "features": list(X.columns)},
        best_model_root
    )
    print(f"[save_model] best_model.pkl (root): {best_model_root.resolve()}")

    # === Guardar en artifacts (NO en raíz) ===
    # Copia del modelo y report (opcional, útil para auditoría)
    try:
        joblib.dump(
            {"pipeline": best_pipe, "num_cols": num_cols, "cat_cols": cat_cols, "features": list(X.columns)},
            outdir / "best_model.pkl"
        )
        report.to_csv(outdir / "model_report.csv", index=False)
    except Exception as e:
        print(f"[warn] no se pudo escribir copias en artifacts: {e}")

    # metrics.json SOLO en artifacts/ (no en raíz)
    summary = {
        "best_model": "XGBoost(train API, bestparams+features)",
        "metrics": report.iloc[0].to_dict(),
        "all_models": report_rows,
        # referencias útiles al CSV limpio
        "clean_csv_relative_to_repo_root": "dataset_clean.csv",
        "clean_csv_in_artifacts": "artifacts/dataset_clean.csv",
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save_metrics] metrics.json: {(outdir / 'metrics.json').resolve()}")

    print("\n✅ Artefactos listos")
    print(f"   (root)      - dataset_clean.csv")
    print(f"   (root)      - best_model.pkl")
    print(f"   (artifacts) - metrics.json, best_model.pkl (copia), model_report.csv, dataset_clean.csv (copia)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo XGBoost para predicción de precio de autos usados.")
    parser.add_argument("--data", type=str, default="used_cars.csv")
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
