# app.py
import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import shap

# ---------------------- Config ----------------------
st.set_page_config(page_title="Precios Justos de Autos Usados", page_icon="🚗", layout="wide")

# Paths: raíz (modelo y data limpia) + artifacts (solo métricas)
APP_ROOT = Path(__file__).resolve().parent
MODEL_PKL = APP_ROOT / "best_model.pkl"                 # ← raíz
DATASET_CSV_DEFAULT = APP_ROOT / "dataset_clean.csv"    # ← raíz
METRICS_JSON = APP_ROOT / "artifacts" / "metrics.json"  # ← artifacts

# --------- Clase wrapper para deserializar el pipeline guardado ---------
class TrainedXGBRegressor:
    """Wrapper mínimo para usar un xgboost.Booster dentro de un Pipeline sklearn."""
    def __init__(self, booster, best_iteration=None):
        self.booster = booster
        self.best_iteration = best_iteration
    def predict(self, X):
        it_range = (0, int(self.best_iteration) + 1) if self.best_iteration is not None else None
        return self.booster.inplace_predict(X, iteration_range=it_range)  # devuelve log1p(price)

# ---------------------- Utilidades (mismas ideas que training.py) ----------------------
def clean_numeric_like(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r'^\s*([0-9]+)\s*k\s*$', lambda m: str(int(m.group(1)) * 1000), regex=True)
    s = (s
         .str.replace(r'[\$,€£]', '', regex=True)
         .str.replace(r'(mi|miles|km|kms|kilometros|kilómetros)', '', regex=True)
         .str.replace(r'[^\d\.-]', '', regex=True)
         .str.replace(r'\s+', '', regex=True))
    return pd.to_numeric(s, errors='coerce')

def normalize_bool_series(s: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False, "si": True, "sí": True}
    return s.astype(str).str.strip().str.lower().map(mapping)

def parse_engine_to_liters(val: str):
    if val is None:
        return np.nan
    s = str(val).lower().strip()
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*l\b', s)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r'(\d{3,5})\s*cc\b', s)
    if m:
        return float(m.group(1)) / 1000.0
    return np.nan

def canonize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """Mapea alias -> nombres estándar y normaliza tipos básicos, para que el app use la misma 'forma' que el training."""
    df = df_in.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    alias = {
        "price": ["price","precio","listing_price","price_usd","precio_usd"],
        "model_year": ["model_year","year","modelyear","año","ano","model year"],
        "mileage": ["mileage","milage","miles","kilometraje","km","odometer"],
        "accident": ["accident","accident_history","has_accident","accidents"],
        "clean_title": ["clean_title","clean title","title_status","clean-title","title"],
        "engine": ["engine","motor","engine_size"],
    }
    def first_present(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    rename_map = {}
    for std, alist in alias.items():
        c = first_present(alist)
        if c and c != std:
            rename_map[c] = std
    if rename_map:
        df = df.rename(columns=rename_map)

    # numéricos clave (para CSV crudo subido por el usuario)
    if "price" in df.columns and "_listed_price" not in df.columns:
        df["_listed_price"] = clean_numeric_like(df["price"])
    if "mileage" in df.columns:
        df["mileage"] = clean_numeric_like(df["mileage"])
    if "model_year" in df.columns:
        df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")

    # booleanos
    if "accident" in df.columns:
        df["accident"] = normalize_bool_series(df["accident"]).fillna(False)
    if "clean_title" in df.columns:
        df["clean_title"] = normalize_bool_series(df["clean_title"]).fillna(False)

    # derivadas
    now_y = datetime.now().year
    if "model_year" in df.columns:
        df["age"] = (now_y - df["model_year"]).clip(lower=0)
    if "mileage" in df.columns:
        df["mileage_per_year"] = df["mileage"] / df.get("age", pd.Series(1, index=df.index)).replace(0, 1)
    if "engine" in df.columns and "engine_liters" not in df.columns:
        df["engine_liters"] = df["engine"].apply(parse_engine_to_liters)

    return df

def add_derived_features_for_row(df_row: pd.DataFrame) -> pd.DataFrame:
    return canonize_columns(df_row)

def ensure_training_columns(df_like: pd.DataFrame, train_features) -> pd.DataFrame:
    if train_features is None:
        return df_like
    return df_like.reindex(columns=list(train_features), fill_value=np.nan)

def price_intervals(pred, mae, rmse, mode="model"):
    # Si no hay métricas, usa ±10%
    if mode == "model" and (mae is None or rmse is None or np.isnan(mae) or np.isnan(rmse)):
        mode = "pct10"
    if mode == "pct10":
        band_low = max(0.0, pred * 0.9);  band_high = pred * 1.1
        int95_low = max(0.0, pred * 0.8); int95_high = pred * 1.2
    else:
        band_low = max(0.0, pred - mae);  band_high = pred + mae
        int95_low = max(0.0, pred - 1.96 * rmse); int95_high = pred + 1.96 * rmse
    return (band_low, band_high), (int95_low, int95_high)

def plot_prediction_with_bands(pred, band, int95):
    x = ["Precio estimado"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[pred], name="Estimado",
                         text=[f"${pred:,.0f}"], textposition="outside"))
    fig.add_shape(type="rect", x0=-0.5, x1=0.5, y0=band[0], y1=band[1],
                  fillcolor="rgba(0, 150, 255, 0.2)", line_width=0, layer="below")
    # intervalo amplio visible, sin texto adicional
    fig.add_shape(type="rect", x0=-0.5, x1=0.5, y0=int95[0], y1=int95[1],
                  fillcolor="rgba(255,165,0,0.15)", line_width=0, layer="below")
    fig.update_layout(height=340, showlegend=False, yaxis_title="USD")
    return fig

# --- Predicción en USD (inverso de log1p) ---
def predict_usd(pipeline, Xlike):
    y_log = pipeline.predict(Xlike)
    return np.expm1(y_log)

# --- Comparables por PRECIO (sin “precio justo”) ---
def compute_similars_by_price(df_base, target_price, brand=None, model=None, pct=0.15, topn=6):
    """Devuelve filas con precio (price) dentro de ±pct del valor objetivo."""
    if df_base is None or df_base.empty or target_price is None or target_price <= 0:
        return None
    df = df_base.copy()
    if brand:
        df = df[df["brand"].astype(str).str.lower() == str(brand).lower()]
    if model is not None:
        same = df["model"].astype(str).str.lower() == str(model).lower()
        if same.sum() >= topn:
            df = df[same]
    if "price" not in df.columns:
        return None
    low, high = target_price * (1 - pct), target_price * (1 + pct)
    df = df[(df["price"] >= low) & (df["price"] <= high)].copy()
    if df.empty:
        return None
    df["_dist_price"] = (df["price"] - target_price).abs()
    return df.sort_values("_dist_price").drop(columns=["_dist_price"]).head(topn)

# ---------- Traducción de columnas para mostrar (nombres y unidades) ----------
DISPLAY_MAP = {
    "brand": "Marca",
    "model": "Modelo",
    "model_year": "Año",
    "mileage": "Kilometraje (km/mi)",
    "fuel_type": "Combustible",
    "engine": "Motor",
    "engine_liters": "Cilindrada (L)",
    "transmission": "Transmisión",
    "ext_col": "Color exterior",
    "int_col": "Color interior",
    "accident": "¿Tuvo accidentes?",
    "clean_title": "Título limpio",
    "age": "Edad (años)",
    "mileage_per_year": "Uso anual (km/mi/año)",
    "price": "Precio ($)",
}

FEATURE_PRETTY = {
    "brand": "Marca",
    "model": "Modelo",
    "model_year": "Año",
    "mileage": "Kilometraje (km/mi)",
    "fuel_type": "Combustible",
    "engine": "Motor",
    "engine_liters": "Cilindrada (L)",
    "transmission": "Transmisión",
    "ext_col": "Color exterior",
    "int_col": "Color interior",
    "accident": "¿Tuvo accidentes?",
    "clean_title": "Título limpio",
    "age": "Edad (años)",
    "mileage_per_year": "Uso anual (km/mi/año)",
}

def pretty_feature_name(name: str) -> str:
    raw = re.sub(r'^(num__|cat__)', '', name)
    return FEATURE_PRETTY.get(raw, raw)

def translate_for_display(df: pd.DataFrame):
    """Devuelve (df_renombrado, formatos) solo para mostrar en tablas y oculta listed/_listed_price."""
    view = df.copy()

    # Unificar precio: si no hay 'price', usar listed/_listed y luego ocultarlos
    if "price" not in view.columns:
        if "_listed_price" in view.columns:
            view["price"] = view["_listed_price"]
        elif "listed_price" in view.columns:
            view["price"] = view["listed_price"]

    # Eliminar siempre columnas listed/_listed para que NO aparezcan
    for col in ["_listed_price", "listed_price"]:
        if col in view.columns:
            view = view.drop(columns=[col])

    # Booleans como Sí/No
    if "accident" in view.columns:
        view["accident"] = view["accident"].map({True: "Sí", False: "No"}).fillna("")
    if "clean_title" in view.columns:
        view["clean_title"] = view["clean_title"].map({True: "Sí", False: "No"}).fillna("")

    # Renombrar columnas conocidas
    rename = {c: DISPLAY_MAP[c] for c in view.columns if c in DISPLAY_MAP}
    view = view.rename(columns=rename)

    # Formatos de presentación
    fmt = {}
    if "Precio ($)" in view.columns: fmt["Precio ($)"] = " ${:,.2f}"
    if "Kilometraje (km/mi)" in view.columns: fmt["Kilometraje (km/mi)"] = "{:,.0f}"
    if "Cilindrada (L)" in view.columns: fmt["Cilindrada (L)"] = "{:.1f}"
    if "Edad (años)" in view.columns: fmt["Edad (años)"] = "{:,.0f}"
    if "Uso anual (km/mi/año)" in view.columns: fmt["Uso anual (km/mi/año)"] = "{:,.0f}"

    return view, fmt

def show_table(df: pd.DataFrame, height: int = 360):
    view, fmt = translate_for_display(df)
    st.dataframe(view.style.format(fmt), use_container_width=True, height=height)

# ---------------------- Carga artefactos ----------------------
@st.cache_resource
def load_pipeline_and_metrics():
    bundle = joblib.load(MODEL_PKL)
    pipeline = bundle["pipeline"]
    features = bundle.get("features", None)
    metrics = {"metrics": {"R2_test": np.nan, "RMSE_test": np.nan, "MAE_test": np.nan}}
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass
    return pipeline, metrics.get("metrics", metrics), features

@st.cache_resource
def load_reference_dataset(path_csv: Path):
    if not Path(path_csv).exists():
        return None
    raw = pd.read_csv(path_csv)
    return canonize_columns(raw)  # asegura tipos si el CSV viene distinto

pipeline, METR, TRAIN_FEATURES = load_pipeline_and_metrics()
preproc = pipeline.named_steps["prep"]
booster = pipeline.named_steps["model"].booster
feature_names = preproc.get_feature_names_out()
shap_explainer = shap.TreeExplainer(booster)

ref_df = load_reference_dataset(DATASET_CSV_DEFAULT)

# ---------------------- Barra lateral ----------------------
st.sidebar.header("Modo")
mode = st.sidebar.radio("Selecciona un flujo", ["Vendedor", "Comprador"])

# Calidad del modelo (con explicación simple)
try:
    st.sidebar.metric("R² (test)", f"{float(METR.get('R2_test', np.nan)):.2f}")
    st.sidebar.metric("RMSE (USD)", f"{float(METR.get('RMSE_test', np.nan)):,.0f}")
    st.sidebar.metric("MAE (USD)", f"{float(METR.get('MAE_test', np.nan)):,.0f}")
except Exception:
    st.sidebar.metric("R² (test)", "N/A")
    st.sidebar.metric("RMSE (USD)", "N/A")
    st.sidebar.metric("MAE (USD)", "N/A")

st.sidebar.markdown(
    "**¿Cómo leer esto?**\n"
    "- **R²**: 0 a 1. Cuánta variación del precio explica el modelo (más alto, mejor).\n"
    "- **MAE**: error medio en USD (promedio de error por auto). **Menor es mejor**.\n"
    "- **RMSE**: parecido al MAE pero penaliza más errores grandes. **Menor es mejor**."
)

# ---------------------- UI principal ----------------------
st.title("🚗 Plataforma de Precios Justos de Autos Usados")
st.caption("Estimación con banda típica, explicación de factores y comparación por **precio similar**. Las tablas muestran nombres entendibles y unidades (sin columnas listed_price).")

# ============================================================
#                       MODO VENDEDOR
# ============================================================
if mode == "Vendedor":
    st.subheader("Vendedor: estima un **rango de precio** y mira ejemplos de precio similar")

    with st.expander("Completa los datos de tu auto"):
        col1, col2, col3 = st.columns(3)
        brand = col1.text_input("Marca", "Toyota")
        model = col2.text_input("Modelo", "Corolla")
        model_year = col3.number_input("Año", min_value=1980, max_value=datetime.now().year+1, value=2018)

        col4, col5, col6 = st.columns(3)
        mileage = col4.number_input("Kilometraje (mi/km)", min_value=0, value=60000)
        fuel_type = col5.text_input("Combustible", "Gasolina")
        engine = col6.text_input("Motor (ej. 2.0L, V6 3.5L, 1500 cc)", "2.0L")

        col7, col8, col9 = st.columns(3)
        transmission = col7.text_input("Transmisión", "Automática")
        ext_col = col8.text_input("Color exterior", "Blanco")
        int_col = col9.text_input("Color interior", "Negro")

        col10, col11 = st.columns(2)
        accident_sel = col10.selectbox("¿Tuvo accidentes?", ["No", "Sí"])
        clean_sel = col11.selectbox("¿Título limpio?", ["Sí", "No"])

        asking_price = st.number_input("Tu precio deseado (opcional, USD)", min_value=0, value=0)
        band_mode = st.radio("¿Cómo mostrar el rango?", ["Por desempeño del modelo (±MAE, ±1.96·RMSE)", "Fijo (±10%)"])

    if st.button("Calcular precio", type="primary", use_container_width=True):
        accident_val = "yes" if str(accident_sel).strip().lower().startswith("s") else "no"
        clean_val = "yes" if str(clean_sel).strip().lower().startswith("s") else "no"

        row_raw = pd.DataFrame([{
            "brand": brand, "model": model, "model_year": int(model_year),
            "mileage": mileage, "fuel_type": fuel_type, "engine": engine,
            "transmission": transmission, "ext_col": ext_col, "int_col": int_col,
            "accident": accident_val, "clean_title": clean_val
        }])

        row = add_derived_features_for_row(row_raw)
        row = ensure_training_columns(row, TRAIN_FEATURES)

        # Predicción (USD reales) SOLO para el cálculo del rango del vendedor
        pred = float(predict_usd(pipeline, row)[0])

        bmode = "model" if band_mode.startswith("Por desempeño") else "pct10"
        band, int95 = price_intervals(pred, METR.get("MAE_test", np.nan), METR.get("RMSE_test", np.nan), mode=bmode)

        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown(f"### 💰 Precio estimado: **${pred:,.0f}**")
            fig = plot_prediction_with_bands(pred, band, int95)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"- **Banda típica (aceptable)**: **${band[0]:,.0f} – ${band[1]:,.0f}**")

            if asking_price and asking_price > 0:
                delta = asking_price - pred
                pct = 100.0 * delta / max(pred, 1.0)
                st.markdown(f"**Tu precio deseado:** ${asking_price:,.0f} → diferencia vs estimado: **{delta:+,.0f}** USD ({pct:+.1f}%).")

        with c2:
            st.markdown("### 🔎 ¿Por qué vale eso? (explicación)")
            X_tr = preproc.transform(row)
            # SHAP en escala log; convertir a USD aprox multiplicando por exp(x) ≈ (pred + 1)
            sv_log = np.array(shap_explainer.shap_values(X_tr)).reshape(-1)
            contrib_usd = sv_log * (pred + 1.0)
            df_contrib = pd.DataFrame({
                "feature": [pretty_feature_name(f) for f in feature_names],
                "contribution_usd": contrib_usd,
                "abs_contrib": np.abs(contrib_usd)
            }).sort_values("abs_contrib", ascending=False).head(8)
            fig2 = go.Figure(go.Bar(
                x=df_contrib["contribution_usd"], y=df_contrib["feature"], orientation="h",
                marker_color=["#2ca02c" if v>=0 else "#d62728" for v in df_contrib["contribution_usd"]],
                hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"
            ))
            fig2.update_layout(height=360, title="Factores que más influyen (± USD)")
            st.plotly_chart(fig2, use_container_width=True)

        # Comparación por PRECIO similar (sin “precio justo”)
        st.markdown("### 🧭 Comparación con **vehículos de precio similar**")
        objetivo = asking_price if asking_price and asking_price > 0 else pred
        comps = compute_similars_by_price(ref_df, objetivo, brand=brand, model=model, pct=0.15, topn=6)
        if comps is None or comps.empty:
            st.info("No se encontraron comparables por precio en la base actual. Prueba con otra combinación o carga un CSV en la sección de Comprador.")
        else:
            show_table(comps, height=360)

# ============================================================
#                       MODO COMPRADOR
# ============================================================
else:
    st.subheader("Comprador: filtra la base y mira **autos por precio** (sin recalcular nada)")
    st.caption("Usa la base limpia del modelo o sube tu CSV. Las tablas muestran nombres entendibles y unidades (sin columnas listed_price).")

    use_uploaded = st.checkbox("Usar mi propio CSV (en vez del dataset por defecto)", value=False)
    if use_uploaded:
        up = st.file_uploader("Sube CSV con columnas similares a entrenamiento", type=["csv"])
        if up:
            df = pd.read_csv(up)
            df = canonize_columns(df)   # normalización si el archivo no viene limpio
        else:
            st.stop()
    else:
        df = ref_df
        if df is None:
            st.error("No se encontró el dataset limpio por defecto (dataset_clean.csv). Sube uno manualmente.")
            st.stop()

    # Filtros (sobre la data limpia)
    with st.expander("Filtros"):
        cols_left, cols_right = st.columns(2)
        brands = sorted(df["brand"].dropna().astype(str).unique()) if "brand" in df.columns else []
        models = sorted(df["model"].dropna().astype(str).unique()) if "model" in df.columns else []
        fuel_types = sorted(df["fuel_type"].dropna().astype(str).unique()) if "fuel_type" in df.columns else []
        transmissions = sorted(df["transmission"].dropna().astype(str).unique()) if "transmission" in df.columns else []

        sel_brand = cols_left.multiselect("Marca", brands)
        sel_model = cols_left.multiselect("Modelo", models)
        if "model_year" in df.columns and df["model_year"].notna().any():
            year_min, year_max = int(df["model_year"].min()), int(df["model_year"].max())
            sel_years = cols_left.slider("Año", min_value=year_min, max_value=year_max, value=(year_min, year_max))
        else:
            sel_years = (None, None)

        sel_fuel = cols_right.multiselect("Combustible", fuel_types)
        sel_trans = cols_right.multiselect("Transmisión", transmissions)
        if "price" in df.columns and df["price"].notna().any():
            pmin, pmax = int(df["price"].min()), int(df["price"].max())
            sel_price = cols_right.slider("Rango de precio (USD)", min_value=pmin, max_value=pmax, value=(pmin, pmax))
        else:
            sel_price = (None, None)

    mask = pd.Series(True, index=df.index)
    if sel_brand and "brand" in df.columns: mask &= df["brand"].astype(str).isin(sel_brand)
    if sel_model and "model" in df.columns: mask &= df["model"].astype(str).isin(sel_model)
    if sel_years[0] is not None and "model_year" in df.columns: mask &= df["model_year"].between(sel_years[0], sel_years[1])
    if sel_fuel and "fuel_type" in df.columns: mask &= df["fuel_type"].astype(str).isin(sel_fuel)
    if sel_trans and "transmission" in df.columns: mask &= df["transmission"].astype(str).isin(sel_trans)
    if sel_price[0] is not None and "price" in df.columns:
        mask &= df["price"].between(sel_price[0], sel_price[1])

    dff = df[mask].copy()
    if dff.empty:
        st.info("No hay resultados con estos filtros.")
        st.stop()

    # Orden por precio
    if "price" in dff.columns and dff["price"].notna().any():
        dff_sorted = dff.sort_values("price", ascending=True)
    else:
        dff_sorted = dff.copy()

    st.markdown("### 🔽 Resultados (data limpia)")
    show_table(dff_sorted, height=520)
