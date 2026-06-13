"""
Tab 2 - CMIP6 (NEX-GDDP-CMIP6) Precipitation Extraction via Google Earth Engine

Streamlit page (multi-page app). Uses a GEE service account stored in
st.secrets["gee_service_account"] to authenticate without browser login.
"""

import io
import json
import time

import ee
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
ALL_MODELS = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CESM2", "CESM2-WACCM",
    "CMCC-CM2-SR5", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3",
    "EC-Earth3-Veg-LR", "FGOALS-g3", "GFDL-CM4", "GFDL-ESM4", "GISS-E2-1-G",
    "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "IITM-ESM", "INM-CM4-8", "INM-CM5-0",
    "IPSL-CM6A-LR", "KACE-1-0-G", "KIOST-ESM", "MIROC-ES2L", "MIROC6",
    "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NESM3", "NorESM2-LM",
    "NorESM2-MM", "TaiESM1", "UKESM1-0-LL",
]

PR_UNIT_TO_MM_DAY = 86400.0  # kg/m2/s -> mm/day
MAX_MODELS_PER_RUN = 5  # keep runtime manageable on Streamlit Cloud


# ============================================================
# EE INIT (service account from st.secrets)
# ============================================================
@st.cache_resource(show_spinner=False)
def init_earth_engine():
    if "gee_service_account" not in st.secrets:
        raise RuntimeError(
            "No 'gee_service_account' found in st.secrets. "
            "Add the service account JSON fields under [gee_service_account] in app secrets."
        )
    # ensure everything is plain str (st.secrets values can be AttrDict-wrapped)
    sa_info = {k: str(v) for k, v in dict(st.secrets["gee_service_account"]).items()}
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    ee.Initialize(credentials, project=sa_info.get("project_id"))
    return True


# ============================================================
# EXTRACTION
# ============================================================
def extract_model_chunk(model, scenario, start_date, end_date, station_fc):
    coll = (ee.ImageCollection("NASA/GDDP-CMIP6")
            .filter(ee.Filter.eq("model", model))
            .filter(ee.Filter.eq("scenario", scenario))
            .filterDate(start_date, end_date)
            .select("pr"))

    def reduce_image(img):
        date_str = img.date().format("YYYY-MM-dd")
        reduced = img.reduceRegions(collection=station_fc, reducer=ee.Reducer.mean(), scale=27830)
        return reduced.map(lambda f: f.set("date", date_str))

    flat = coll.map(reduce_image).flatten()
    flat = flat.select(["date", "Station_ID", "mean"])
    return flat.getInfo()


def extract_model(model, scenario, start_year, end_year, station_fc, chunk_years, progress_cb=None):
    all_records = []
    years = list(range(start_year, end_year + 1, chunk_years))
    total_chunks = len(years)

    for idx, year in enumerate(years):
        chunk_end = min(year + chunk_years - 1, end_year)
        start_date, end_date = f"{year}-01-01", f"{chunk_end}-12-31"

        for attempt in range(3):
            try:
                result = extract_model_chunk(model, scenario, start_date, end_date, station_fc)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"{model} {start_date}-{end_date} failed after retries: {e}")
                time.sleep(5)

        for feat in result["features"]:
            props = feat["properties"]
            all_records.append({
                "Date": props["date"],
                "Station_ID": props["Station_ID"],
                "pr_kg_m2_s": props.get("mean"),
            })

        if progress_cb:
            progress_cb((idx + 1) / total_chunks, f"{model}: {start_date} to {end_date}")

    df = pd.DataFrame(all_records)
    if df.empty:
        raise RuntimeError(f"No data returned for model {model}. It may not have global coverage at these points.")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Precip_mm"] = df["pr_kg_m2_s"] * PR_UNIT_TO_MM_DAY
    pivot = df.pivot(index="Date", columns="Station_ID", values="Precip_mm").reset_index()
    return pivot.sort_values("Date").reset_index(drop=True)


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def build_monthly_and_basin(daily_df):
    """From a Date x Station daily DataFrame, build monthly (Year, Month, stations)
    and basin-average monthly (Year, Month, Basin_Rainfall_mm) DataFrames -
    same convention as Tab 1 IMD outputs."""
    station_cols = [c for c in daily_df.columns if c != "Date"]
    d = daily_df.copy()
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    monthly = d.drop(columns="Date").groupby(["Year", "Month"])[station_cols].sum(min_count=1).reset_index()

    basin_monthly = monthly[["Year", "Month"]].copy()
    basin_monthly["Basin_Rainfall_mm"] = monthly[station_cols].mean(axis=1)

    return monthly, basin_monthly


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="CMIP6 Downscaled Precipitation Extraction", layout="wide")
st.title("CMIP6 (NEX-GDDP-CMIP6) Bias-Corrected, Downscaled Rainfall Extraction")
st.caption("MODEL 1 — Tab 2: Download NASA NEX-GDDP-CMIP6 bias-corrected, statistically downscaled "
           "historical precipitation for the representative stations, converted to mm/day, "
           "same Date x Station format as IMD for R2/NSE comparison.")

st.subheader("1. Stations")
station_input_mode = st.radio("Provide station locations", ["Upload Final_Stations.xlsx", "Enter manually"])

stations_df = None
if station_input_mode == "Upload Final_Stations.xlsx":
    f = st.file_uploader("Final_Stations.xlsx (from Tab 1)", type=["xlsx"])
    if f is not None:
        raw = pd.read_excel(f)
        # case-insensitive column matching
        col_map = {c.lower(): c for c in raw.columns}
        required = ["station_id", "latitude", "longitude"]
        missing = [r for r in required if r not in col_map]
        if missing:
            st.error(
                f"Uploaded file is missing required column(s): {missing}. "
                f"Found columns: {list(raw.columns)}. "
                f"Make sure you uploaded Final_Stations.xlsx (with Station_ID, Latitude, Longitude columns), "
                f"not the rainfall data file."
            )
        else:
            stations_df = raw[[col_map["station_id"], col_map["latitude"], col_map["longitude"]]]
            stations_df.columns = ["Station_ID", "Latitude", "Longitude"]
            st.dataframe(stations_df, hide_index=True)
else:
    default_text = "Station_ID,Latitude,Longitude\nST001,18.50,73.75\nST002,17.00,77.00"
    text = st.text_area("CSV: Station_ID,Latitude,Longitude", value=default_text, height=150)
    try:
        stations_df = pd.read_csv(io.StringIO(text))
        st.dataframe(stations_df, hide_index=True)
    except Exception as e:
        st.error(f"Could not parse station list: {e}")

st.subheader("2. Models & period")
selected_models = st.multiselect(
    f"GCM models (max {MAX_MODELS_PER_RUN} per run)", ALL_MODELS,
    default=["MPI-ESM1-2-HR", "EC-Earth3", "MIROC6"]
)
col1, col2 = st.columns(2)
start_year = col1.number_input("Start year", value=1984, min_value=1950, max_value=2014)
end_year = col2.number_input("End year", value=2014, min_value=1950, max_value=2014)
chunk_years = 1  # hardcoded - larger chunks were timing out

if len(selected_models) > MAX_MODELS_PER_RUN:
    st.warning(f"Please select at most {MAX_MODELS_PER_RUN} models per run (Streamlit Cloud resource limits). "
               f"Run remaining models in a separate run.")

run_disabled = (
    stations_df is None or len(selected_models) == 0 or len(selected_models) > MAX_MODELS_PER_RUN
)
run_btn = st.button("Extract CMIP6 data", type="primary", disabled=run_disabled)

if run_btn:
    with st.spinner("Connecting to Earth Engine..."):
        try:
            init_earth_engine()
        except Exception as e:
            st.error(f"Earth Engine init failed: {e}")
            st.stop()

    station_features = [
        ee.Feature(ee.Geometry.Point([row["Longitude"], row["Latitude"]]), {"Station_ID": row["Station_ID"]})
        for _, row in stations_df.iterrows()
    ]
    station_fc = ee.FeatureCollection(station_features)

    model_outputs = {}
    for model in selected_models:
        st.write(f"**Extracting {model}**")
        progress = st.progress(0.0)
        status = st.empty()
        try:
            df_model = extract_model(
                model, "historical", int(start_year), int(end_year), station_fc, int(chunk_years),
                progress_cb=lambda frac, msg: (progress.progress(frac), status.write(msg))
            )
            model_outputs[model] = df_model
            progress.progress(1.0)
            status.write(f"Done: {df_model.shape[0]} days x {df_model.shape[1]-1} stations")
        except Exception as e:
            st.error(f"{model} failed: {e}")

    if model_outputs:
        st.subheader("3. Results & downloads")
        st.caption("Each model is provided in the same daily / monthly / basin-monthly formats as the "
                   "IMD outputs from Tab 1 (Representative_Stations_Rainfall, Representative_Stations_Monthly, "
                   "Basin_Monthly_Rainfall) - so they can be combined with IMD files for Tab 3 evaluation.")
        for model, df_model in model_outputs.items():
            monthly, basin_monthly = build_monthly_and_basin(df_model)
            with st.expander(f"{model} — {df_model.shape[0]} days"):
                st.dataframe(df_model.head(), hide_index=True)
                base = f"CMIP6_{model}_{start_year}-{end_year}"
                c1, c2, c3 = st.columns(3)
                c1.download_button(f"Download {base}_daily.xlsx", to_excel_bytes(df_model), f"{base}_daily.xlsx")
                c2.download_button(f"Download {base}_monthly.xlsx", to_excel_bytes(monthly), f"{base}_monthly.xlsx")
                c3.download_button(f"Download {base}_basin_monthly.xlsx", to_excel_bytes(basin_monthly), f"{base}_basin_monthly.xlsx")

st.markdown("---")
st.caption("App developed by Anandita Raj and Prof. Raj Mohan Singh")
