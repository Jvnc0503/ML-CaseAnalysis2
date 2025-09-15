# app.py
import os
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

ARTIFACTS_DIR = Path("artifacts")
MODEL_PKL = ARTIFACTS_DIR / "best_model.pkl"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"

# CSV base (misma estructura que entrenaste); úsalo para: comprador (navegar) y comps del vendedor.
# Si quieres usar otro archivo, cámbialo aquí o deja que el comprador suba el suyo.
DATASET_CSV_DEFAULT = "used_cars.csv"

class TrainedXGBRegressor:
    """Wrapper mínimo para usar un xgboost.Booster dentro de un Pipeline sklearn."""
    def __init__(self, booster, best_iteration=None):
        self.booster = booster
        self.best_iteration = best_iteration

    def predict(self, X):
        it_range = (0, int(self.best_iteration) + 1) if self.best_iteration is not None else None
        # X llega ya transformado por el preprocesador del Pipeline
        import numpy as _np
        return self.booster.inplace_predict(X, iteration_range=it_range)

# ---------------------- Carga artefactos ----------------------
@st.cache_resource
def load_pipeline_and_metrics():
    bundle = joblib.load(MODEL_PKL)
    pipeline = bundle["pipeline"]
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return pipeline, metrics["metrics"]

@st.cache_resource
def load_reference_dataset(path_csv: str):
    if not Path(path_csv).exists():
        return None
    df = pd.read_csv(path_csv)
    # normaliza columna 'price' por si viene con símbolos
    if "price" in df.columns:
        df["_listed_price"] = clean_numeric_like(df["price"])
    return df

# ---------------------- Utilidades ----------------------
def clean_numeric_like(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r'^\s*([0-9]+)\s*k\s*$', lambda m: str(int(m.group(1)) * 1000), regex=True)
    s = (s
         .str.replace(r'[\$,€£]', '', regex=True)
         .str.replace(r'(mi|miles|km|kms|kilometros|kilómetros)', '', regex=True)
         .str.replace(r'[^\d\.-]', '', regex=True)
         .str.replace(r'\s+', '', regex=True))
    return pd.to_numeric(s, errors='coerce')

def price_intervals(pred, mae, rmse, mode="model"):
    """
    mode="model" -> banda típica ±MAE y banda amplia ±1.96·RMSE
    mode="pct10" -> banda fija ±10% del estimado
    """
    if mode == "pct10":
        band_low = max(0.0, pred * 0.9)
        band_high = pred * 1.1
        int95_low = max(0.0, pred * 0.8)
        int95_high = pred * 1.2
    else:
        band_low = max(0.0, pred - mae)
        band_high = pred + mae
        int95_low = max(0.0, pred - 1.96 * rmse)
        int95_high = pred + 1.96 * rmse
    return (band_low, band_high), (int95_low, int95_high)

def plot_prediction_with_bands(pred, band, int95):
    x = ["Precio estimado"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[pred], name="Estimado",
                         text=[f"${pred:,.0f}"], textposition="outside"))
    # bandas
    fig.add_shape(type="rect", x0=-0.5, x1=0.5, y0=band[0], y1=band[1],
                  fillcolor="rgba(0, 150, 255, 0.2)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-0.5, x1=0.5, y0=int95[0], y1=int95[1],
                  fillcolor="rgba(255,165,0,0.15)", line_width=0, layer="below")
    fig.update_layout(height=340, showlegend=False, yaxis_title="USD")
    return fig

def identify_outlier_single(asking, pred, mae, rmse, k_mae=2.0, k_rmse=1.0):
    """ True si |asking - pred| > max(k*MAE, k_rmse*RMSE) """
    thresh = max(k_mae * mae, k_rmse * rmse)
    flag = abs(asking - pred) > thresh
    return flag, thresh

def identify_outliers_vector(prices, preds, mae, rmse, k_mae=2.0, k_rmse=1.0):
    residual = prices - preds
    thresh = np.maximum(k_mae * mae, k_rmse * rmse)
    flags = np.abs(residual) > thresh
    return flags, residual, thresh

def compute_similars(df_base, query_row, topn=5):
    """
    Comparación con vehículos similares:
    - misma marca y modelo si es posible,
    - si no hay suficientes, misma marca,
    - y filtra por año ±2 y kilometraje ±20%.
    Ordena por distancia simple (|delta_year| + |delta_mileage_pct|).
    """
    if df_base is None:
        return None
    df = df_base.copy()
    needed = ["brand", "model", "model_year", "mileage"]
    if not all(c in df.columns for c in needed):
        return None

    # Filtros por categoría
    mask = pd.Series(True, index=df.index)
    if "brand" in df.columns and pd.notna(query_row.get("brand")):
        mask &= (df["brand"].astype(str).str.lower() == str(query_row["brand"]).lower())
    if "model" in df.columns and pd.notna(query_row.get("model")):
        same_model = (df["model"].astype(str).str.lower() == str(query_row["model"]).lower())
        # si hay suficientes por modelo, usa modelo; si no, deja solo marca
        if same_model.sum() >= topn:
            mask &= same_model

    df = df[mask].copy()
    if df.empty:
        return None

    # aproximación por rango
    y = float(query_row.get("model_year", np.nan))
    m = float(query_row.get("mileage", np.nan))
    if not np.isnan(y):
        df = df[df["model_year"].between(y - 2, y + 2)]
    if not np.isnan(m) and m > 0:
        df = df[(df["mileage"] >= 0.8 * m) & (df["mileage"] <= 1.2 * m)]

    if df.empty:
        return None

    # distancia simple
    df["_dist"] = 0.0
    if not np.isnan(y):
        df["_dist"] += (df["model_year"] - y).abs()
    if not np.isnan(m) and m > 0:
        df["_dist"] += (df["mileage"] / m - 1.0).abs()
    df = df.sort_values("_dist").head(topn)
    return df

def format_money(df, cols):
    return df.style.format({c: " ${:,.0f}" for c in cols if c in df.columns})

# ---------------------- Carga ----------------------
pipeline, METR = load_pipeline_and_metrics()
preproc = pipeline.named_steps["prep"]
booster = pipeline.named_steps["model"].booster
feature_names = preproc.get_feature_names_out()
shap_explainer = shap.TreeExplainer(booster)

ref_df = load_reference_dataset(DATASET_CSV_DEFAULT)

# ---------------------- Barra lateral ----------------------
st.sidebar.header("Modo")
mode = st.sidebar.radio("Selecciona un flujo", ["Vendedor", "Comprador"])

st.sidebar.header("Calidad del modelo")
st.sidebar.metric("R² (test)", f"{METR['R2_test']:.2f}")
st.sidebar.metric("RMSE (USD)", f"{METR['RMSE_test']:,.0f}")
st.sidebar.metric("MAE (USD)", f"{METR['MAE_test']:,.0f}")
st.sidebar.caption("RMSE: error cuadrático medio raíz; MAE: error absoluto medio.")

# ---------------------- UI principal ----------------------
st.title("🚗 Plataforma de Precios Justos de Autos Usados")
st.caption("Transparencia y confianza: estimación con rangos, explicación de factores, comparación con similares y alertas para precios inusuales.")

# ============================================================
#                       MODO VENDEDOR
# ============================================================
if mode == "Vendedor":
    st.subheader("Vendedor: estima un **rango de precio justo** y compáralo con tu expectativa")

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
        accident = col10.selectbox("¿Tuvo accidentes?", ["No", "Sí"])
        clean_title = col11.selectbox("¿Título limpio?", ["Sí", "No"])

        asking_price = st.number_input("Tu precio deseado (opcional, USD)", min_value=0, value=0)
        band_mode = st.radio("¿Cómo mostrar el rango?", ["Por desempeño del modelo (±MAE, ±1.96·RMSE)", "Fijo (±10%)"])

    if st.button("Calcular precio", type="primary", use_container_width=True):
        # Construir fila cruda para el pipeline
        row = pd.DataFrame([{
            "brand": brand, "model": model, "model_year": int(model_year),
            "mileage": mileage, "fuel_type": fuel_type, "engine": engine,
            "transmission": transmission, "ext_col": ext_col, "int_col": int_col,
            "accident": accident, "clean_title": clean_title
        }])

        # Predicción y bandas
        pred = float(pipeline.predict(row)[0])
        bmode = "model" if band_mode.startswith("Por desempeño") else "pct10"
        band, int95 = price_intervals(pred, METR["MAE_test"], METR["RMSE_test"], mode=bmode)

        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown(f"### 💰 Precio estimado: **${pred:,.0f}**")
            st.caption("Incluye bandas de incertidumbre. Azul: banda típica; Naranja: intervalo amplio.")
            fig = plot_prediction_with_bands(pred, band, int95)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"- Banda típica: **${band[0]:,.0f} – ${band[1]:,.0f}**")
            st.markdown(f"- Intervalo amplio: **${int95[0]:,.0f} – ${int95[1]:,.0f}**")

            # comparar con expectativa del vendedor
            if asking_price and asking_price > 0:
                delta = asking_price - pred
                pct = 100.0 * delta / max(pred, 1.0)
                st.markdown(f"**Tu precio deseado:** ${asking_price:,.0f} → diferencia: **{delta:+,.0f}** USD ({pct:+.1f}%).")
                flag, thresh = identify_outlier_single(asking_price, pred, METR["MAE_test"], METR["RMSE_test"])
                if flag:
                    st.warning(f"⚠️ Precio inusual (|diferencia| > ~{thresh:,.0f} USD). Revisa fotos, mantenimiento y documentación. Podrías recibir menos interés.")
                else:
                    st.info("Tu precio está alineado con el mercado para autos similares.")

        with c2:
            # Explicabilidad
            st.markdown("### 🔎 ¿Por qué vale eso? (explicación)")
            X_tr = preproc.transform(row)
            sv = shap.TreeExplainer(booster).shap_values(X_tr)
            sv = np.array(sv).reshape(-1)
            # aproximación de contribuciones en USD
            contrib_usd = sv * max(pred, 1.0)
            df_contrib = pd.DataFrame({
                "feature": feature_names,
                "contribution_usd": contrib_usd,
                "abs_contrib": np.abs(contrib_usd)
            }).sort_values("abs_contrib", ascending=False).head(8)
            df_contrib["feature"] = df_contrib["feature"].str.replace("^num__", "", regex=True).str.replace("^cat__", "", regex=True)
            fig2 = go.Figure(go.Bar(
                x=df_contrib["contribution_usd"], y=df_contrib["feature"], orientation="h",
                marker_color=["#2ca02c" if v>=0 else "#d62728" for v in df_contrib["contribution_usd"]],
                hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"
            ))
            fig2.update_layout(height=360, title="Factores que más influyen (± USD)")
            st.plotly_chart(fig2, use_container_width=True)

        # Comparación con similares desde el dataset base
        st.markdown("### 🧭 Comparación con **vehículos similares**")
        comps = compute_similars(ref_df, {
            "brand": brand, "model": model, "model_year": model_year, "mileage": mileage
        }, topn=6)
        if comps is None or comps.empty:
            st.info("No se encontraron comparables en la base actual. Prueba con otra combinación o carga un CSV en la sección de Comprador.")
        else:
            # estimar precio justo de comps
            comps = comps.copy()
            preds_comps = pipeline.predict(comps)
            comps["_fair_price"] = preds_comps
            if "_listed_price" not in comps.columns and "price" in comps.columns:
                comps["_listed_price"] = clean_numeric_like(comps["price"])
            comps["_gap_usd"] = comps["_fair_price"] - comps.get("_listed_price", np.nan)
            show = ["brand","model","model_year","mileage","_listed_price","_fair_price","_gap_usd"]
            show = [c for c in show if c in comps.columns]
            st.dataframe(format_money(comps[show].rename(columns={
                "_listed_price":"listed_price","_fair_price":"fair_price","_gap_usd":"gap_usd"
            }), ["listed_price","fair_price","gap_usd"]), use_container_width=True, height=340)

    st.info("💡 Recomendación: si tu precio deseado está fuera del rango, considera ajustar o reforzar la publicación con evidencia (mantenimiento, histórico, fotos claras).")

# ============================================================
#                       MODO COMPRADOR
# ============================================================
else:
    st.subheader("Comprador: filtra la base y mira **mejores precios** y **mejores ofertas**")
    st.caption("Usa la misma base del modelo o sube tu CSV. Filtra por etiquetas y ordena por precio más bajo o por mejor descuento vs precio justo.")

    # Fuente de datos: base por defecto o subida
    c1, c2 = st.columns([2,1])
    with c1:
        use_uploaded = st.checkbox("Usar mi propio CSV (en vez del dataset por defecto)", value=False)
    if use_uploaded:
        up = st.file_uploader("Sube CSV con columnas similares a entrenamiento", type=["csv"])
        if up:
            df = pd.read_csv(up)
        else:
            st.stop()
    else:
        df = ref_df
        if df is None:
            st.error("No se encontró el dataset por defecto. Sube uno manualmente.")
            st.stop()

    # Normaliza precio listado
    if "price" in df.columns and "_listed_price" not in df.columns:
        df["_listed_price"] = clean_numeric_like(df["price"])

    # Filtros por labels disponibles
    with st.expander("Filtros"):
        cols_left, cols_right = st.columns(2)
        brands = sorted(df["brand"].dropna().astype(str).unique()) if "brand" in df.columns else []
        models = sorted(df["model"].dropna().astype(str).unique()) if "model" in df.columns else []
        fuel_types = sorted(df["fuel_type"].dropna().astype(str).unique()) if "fuel_type" in df.columns else []
        transmissions = sorted(df["transmission"].dropna().astype(str).unique()) if "transmission" in df.columns else []

        sel_brand = cols_left.multiselect("Marca", brands, default=brands[:1] if brands else [])
        sel_model = cols_left.multiselect("Modelo", models, default=models[:2] if models else [])
        year_min, year_max = cols_left.slider("Año", min_value=int(df["model_year"].min()), max_value=int(df["model_year"].max()), value=(2012, 2022)) if "model_year" in df.columns else (None, None)

        sel_fuel = cols_right.multiselect("Combustible", fuel_types)
        sel_trans = cols_right.multiselect("Transmisión", transmissions)
        max_price = cols_right.number_input("Precio listado máximo (USD)", min_value=0, value=int(df["_listed_price"].quantile(0.9)) if "_listed_price" in df.columns else 0)

    mask = pd.Series(True, index=df.index)
    if sel_brand and "brand" in df.columns: mask &= df["brand"].astype(str).isin(sel_brand)
    if sel_model and "model" in df.columns: mask &= df["model"].astype(str).isin(sel_model)
    if year_min is not None and "model_year" in df.columns: mask &= df["model_year"].between(year_min, year_max)
    if sel_fuel and "fuel_type" in df.columns: mask &= df["fuel_type"].astype(str).isin(sel_fuel)
    if sel_trans and "transmission" in df.columns: mask &= df["transmission"].astype(str).isin(sel_trans)
    if max_price and "_listed_price" in df.columns and max_price > 0: mask &= (df["_listed_price"] <= max_price)

    dff = df[mask].copy()
    if dff.empty:
        st.info("No hay resultados con estos filtros.")
        st.stop()

    # Predice precio justo y calcula ofertas
    preds = pipeline.predict(dff)
    dff["_fair_price"] = preds
    dff["_gap_usd"] = dff["_fair_price"] - dff.get("_listed_price", np.nan)
    dff["_deal_pct"] = (dff["_fair_price"] - dff["_listed_price"]) / dff["_fair_price"] * 100.0

    # Alertas de precios inusuales en el subset filtrado
    flags, residuals, thresh = identify_outliers_vector(
        dff["_listed_price"].values, dff["_fair_price"].values,
        METR["MAE_test"], METR["RMSE_test"], k_mae=2.0, k_rmse=1.0
    )
    dff["_alert"] = np.where(flags, "⚠️ Inusual", "")

    st.markdown("### 🔽 Precios **más bajos** (mejor orden por precio listado)")
    if "_listed_price" in dff.columns:
        low = dff.sort_values("_listed_price", ascending=True).head(20)
        show_low = ["brand","model","model_year","mileage","_listed_price","_fair_price","_gap_usd","_deal_pct","_alert"]
        show_low = [c for c in show_low if c in low.columns]
        st.dataframe(
            low[show_low].rename(columns={
                "_listed_price":"listed_price", "_fair_price":"fair_price",
                "_gap_usd":"gap_usd", "_deal_pct":"deal_pct(%)"
            }).style.format({
                "listed_price":" ${:,.0f}", "fair_price":" ${:,.0f}",
                "gap_usd":" ${:,.0f}", "deal_pct(%)":"{:.1f}"
            }),
            use_container_width=True, height=380
        )
    else:
        st.info("El CSV no tiene columna 'price' (precio listado). No se puede ordenar por precio más bajo.")

    st.markdown("### 🏷️ Mejores **ofertas** (mayor descuento vs precio justo)")
    deals = dff.sort_values("_deal_pct", ascending=False).head(20)
    show_deal = ["brand","model","model_year","mileage","_listed_price","_fair_price","_gap_usd","_deal_pct","_alert"]
    show_deal = [c for c in show_deal if c in deals.columns]
    st.dataframe(
        deals[show_deal].rename(columns={
            "_listed_price":"listed_price", "_fair_price":"fair_price",
            "_gap_usd":"gap_usd", "_deal_pct":"deal_pct(%)"
        }).style.format({
            "listed_price":" ${:,.0f}", "fair_price":" ${:,.0f}",
            "gap_usd":" ${:,.0f}", "deal_pct(%)":"{:.1f}"
        }),
        use_container_width=True, height=380
    )

    st.caption(f"Umbral de alerta actual ≈ max(2×MAE, 1×RMSE) = {max(2*METR['MAE_test'], METR['RMSE_test']):,.0f} USD. "
               "Un precio muy alejado puede indicar error, fraude o que faltan datos (daños, siniestralidad, etc.).")
