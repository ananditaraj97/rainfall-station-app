"""
Tab 4 - Future Climate Projections (NEX-GDDP-CMIP6, SSP245 / SSP585)

Same extraction approach as Tab 2 (precipitation + Tmax/Tmin/relative
humidity/wind/solar radiation), but for future scenarios and standard
climate-impact time slices (Near/Mid/Far future).
"""

import io
import time
import zipfile

import ee
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

MAX_MODELS_PER_RUN = 2  # future slices (up to 80yr) x 6 variables - keep small

VARIABLES = {
    "pr":      {"label": "Precipitation",     "unit": "mm/day",     "convert": lambda v: v * 86400.0, "agg": "sum",  "swat_file": "PCP"},
    "tasmax":  {"label": "Max Temperature",   "unit": "degC",       "convert": lambda v: v - 273.15,  "agg": "mean", "swat_file": "TMP (max)"},
    "tasmin":  {"label": "Min Temperature",   "unit": "degC",       "convert": lambda v: v - 273.15,  "agg": "mean", "swat_file": "TMP (min)"},
    "hurs":    {"label": "Relative Humidity", "unit": "%",          "convert": lambda v: v,           "agg": "mean", "swat_file": "HMD"},
    "sfcWind": {"label": "Wind Speed",        "unit": "m/s",        "convert": lambda v: v,           "agg": "mean", "swat_file": "WND"},
    "rsds":    {"label": "Solar Radiation",   "unit": "MJ/m2/day",  "convert": lambda v: v * 0.0864,  "agg": "mean", "swat_file": "SLR"},
}

TIME_SLICES = {
    "Near Future (2021-2040)": (2021, 2040),
    "Mid Future (2041-2070)": (2041, 2070),
    "Far Future (2071-2100)": (2071, 2100),
    "Custom": None,
}

SCENARIOS = ["ssp245", "ssp585"]


# ============================================================
# EE INIT (service account from st.secrets) - same as Tab 2
# ============================================================
@st.cache_resource(show_spinner=False)
def init_earth_engine():
    if "gee_service_account" not in st.secrets:
        raise RuntimeError(
            "No 'gee_service_account' found in st.secrets. "
            "Add the service account JSON fields under [gee_service_account] in app secrets."
        )
    sa_info = {k: str(v) for k, v in dict(st.secrets["gee_service_account"]).items()}
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    ee.Initialize(credentials, project=sa_info.get("project_id"))
    return True


# ============================================================
# EXTRACTION (same mechanics as Tab 2, scenario is a parameter)
# ============================================================
def extract_chunk(model, scenario, start_date, end_date, station_fc, bands):
    coll = (ee.ImageCollection("NASA/GDDP-CMIP6")
            .filter(ee.Filter.eq("model", model))
            .filter(ee.Filter.eq("scenario", scenario))
            .filterDate(start_date, end_date)
            .select(bands))

    def reduce_image(img):
        date_str = img.date().format("YYYY-MM-dd")
        reduced = img.reduceRegions(collection=station_fc, reducer=ee.Reducer.mean(), scale=27830)
        return reduced.map(lambda f: f.set("date", date_str))

    flat = coll.map(reduce_image).flatten()
    flat = flat.select(["date", "Station_ID"] + bands)
    return flat.getInfo()


def extract_model(model, scenario, start_year, end_year, station_fc, bands, chunk_years=1, progress_cb=None):
    """Returns a long DataFrame: Date, Station_ID, <band1>, <band2>, ... (raw values, unconverted)."""
    all_records = []
    years = list(range(start_year, end_year + 1, chunk_years))
    total_chunks = len(years)

    for idx, year in enumerate(years):
        chunk_end = min(year + chunk_years - 1, end_year)
        start_date, end_date = f"{year}-01-01", f"{chunk_end}-12-31"

        for attempt in range(3):
            try:
                result = extract_chunk(model, scenario, start_date, end_date, station_fc, bands)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"{model} {start_date}-{end_date} failed after retries: {e}")
                time.sleep(5)

        for feat in result["features"]:
            props = feat["properties"]
            rec = {"Date": props["date"], "Station_ID": props["Station_ID"]}
            for b in bands:
                rec[b] = props.get(b)
            all_records.append(rec)

        if progress_cb:
            progress_cb((idx + 1) / total_chunks, f"{model}: {start_date} to {end_date}")

    df = pd.DataFrame(all_records)
    if df.empty:
        raise RuntimeError(f"No data returned for model {model}. It may not have global coverage at these points.")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def pivot_variable(long_df, band):
    conv = VARIABLES[band]["convert"]
    d = long_df[["Date", "Station_ID", band]].copy()
    d[band] = conv(d[band].astype(float))
    pivot = d.pivot(index="Date", columns="Station_ID", values=band).reset_index()
    return pivot.sort_values("Date").reset_index(drop=True)


def build_monthly_and_basin(daily_df, agg="sum"):
    station_cols = [c for c in daily_df.columns if c != "Date"]
    d = daily_df.copy()
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    if agg == "sum":
        monthly = d.drop(columns="Date").groupby(["Year", "Month"])[station_cols].sum(min_count=1).reset_index()
    else:
        monthly = d.drop(columns="Date").groupby(["Year", "Month"])[station_cols].mean().reset_index()

    basin_monthly = monthly[["Year", "Month"]].copy()
    basin_monthly["Basin_Value"] = monthly[station_cols].mean(axis=1)
    return monthly, basin_monthly


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Future Climate Projections (CMIP6)", layout="wide")
st.title("Future Climate Projections (NEX-GDDP-CMIP6, SSP245 / SSP585)")
st.caption("MODEL 1 — Tab 4: Download NEX-GDDP-CMIP6 bias-corrected, downscaled future precipitation "
           "AND the remaining SWAT-relevant variables (Tmax, Tmin, relative humidity, wind speed, "
           "solar radiation) for the representative stations under SSP245/SSP585, for standard "
           "climate-impact time slices (Near/Mid/Far future), in the same Date x Station format as Tabs 1-2.")

st.subheader("1. Stations")
station_input_mode = st.radio("Provide station locations", ["Upload Final_Stations.xlsx", "Enter manually"])

stations_df = None
if station_input_mode == "Upload Final_Stations.xlsx":
    f = st.file_uploader("Final_Stations.xlsx (from Tab 1)", type=["xlsx"])
    if f is not None:
        raw = pd.read_excel(f)
        col_map = {c.lower(): c for c in raw.columns}
        required = ["station_id", "latitude", "longitude"]
        missing = [r for r in required if r not in col_map]
        if missing:
            st.error(
                f"Uploaded file is missing required column(s): {missing}. "
                f"Found columns: {list(raw.columns)}. "
                f"Make sure you uploaded Final_Stations.xlsx (with Station_ID, Latitude, Longitude columns)."
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

st.subheader("2. Variables, scenario, time slice & models")
var_options = {f"{k} - {v['label']} ({v['unit']})": k for k, v in VARIABLES.items()}
selected_var_labels = st.multiselect(
    "Variables to extract (default: all 6, for SWAT PCP/TMP/HMD/WND/SLR)",
    list(var_options.keys()), default=list(var_options.keys())
)
selected_vars = [var_options[lbl] for lbl in selected_var_labels]

c1, c2 = st.columns(2)
scenario = c1.selectbox("SSP scenario", SCENARIOS)
slice_choice = c2.selectbox("Time slice", list(TIME_SLICES.keys()), index=0)

if TIME_SLICES[slice_choice] is not None:
    default_start, default_end = TIME_SLICES[slice_choice]
else:
    default_start, default_end = 2021, 2040

c3, c4 = st.columns(2)
start_year = c3.number_input("Start year", value=default_start, min_value=2015, max_value=2100)
end_year = c4.number_input("End year", value=default_end, min_value=2015, max_value=2100)
chunk_years = 1  # hardcoded - more reliable than larger chunks

n_years = int(end_year) - int(start_year) + 1
st.caption(f"{n_years}-year period -> {n_years} Earth Engine request(s) per model.")

selected_models = st.multiselect(
    f"GCM models (max {MAX_MODELS_PER_RUN} per run - future slices need more requests than historical)",
    ALL_MODELS, default=["MPI-ESM1-2-HR"]
)

if len(selected_models) > MAX_MODELS_PER_RUN:
    st.warning(f"Please select at most {MAX_MODELS_PER_RUN} models per run. "
               f"Run remaining models in a separate run.")

run_disabled = (
    stations_df is None or not selected_vars or len(selected_models) == 0 or len(selected_models) > MAX_MODELS_PER_RUN
)
run_btn = st.button("Extract future projections", type="primary", disabled=run_disabled)

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

    model_outputs = {}  # model -> {var: (daily, monthly, basin_monthly)}
    for model in selected_models:
        st.write(f"**Extracting {model} ({scenario}, {start_year}-{end_year})**")
        progress = st.progress(0.0)
        status = st.empty()
        try:
            long_df = extract_model(
                model, scenario, int(start_year), int(end_year), station_fc, selected_vars, chunk_years,
                progress_cb=lambda frac, msg: (progress.progress(frac), status.write(msg))
            )
            var_outputs = {}
            for band in selected_vars:
                daily = pivot_variable(long_df, band)
                monthly, basin_monthly = build_monthly_and_basin(daily, agg=VARIABLES[band]["agg"])
                var_outputs[band] = (daily, monthly, basin_monthly)
            model_outputs[model] = var_outputs
            progress.progress(1.0)
            status.write(f"Done: {long_df['Date'].nunique()} days x {len(stations_df)} stations, "
                         f"{len(selected_vars)} variable(s)")
        except Exception as e:
            st.error(f"{model} failed: {e}")

    if model_outputs:
        st.subheader("3. Results & downloads")
        st.caption("Same daily / monthly / basin-monthly formats as Tab 2, with scenario and time-slice "
                   "in the filename, e.g. CMIP6_<model>_pr_ssp245_2021-2040_basin_monthly.xlsx. "
                   "Download a single zip per model for all variables (use in Tab 5 ensembles and Tab 6 "
                   "SWAT weather files).")

        for model, var_outputs in model_outputs.items():
            with st.expander(f"{model}"):
                preview_var = "pr" if "pr" in var_outputs else selected_vars[0]
                st.dataframe(var_outputs[preview_var][0].head(), hide_index=True)

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for band, (daily, monthly, basin_monthly) in var_outputs.items():
                        base = f"CMIP6_{model}_{band}_{scenario}_{start_year}-{end_year}"
                        zf.writestr(f"{base}_daily.xlsx", to_excel_bytes(daily))
                        zf.writestr(f"{base}_monthly.xlsx", to_excel_bytes(monthly))
                        zf.writestr(f"{base}_basin_monthly.xlsx", to_excel_bytes(basin_monthly))
                zip_buf.seek(0)
                st.download_button(f"Download {model}_{scenario}_{start_year}-{end_year}_all_variables.zip",
                                    zip_buf.getvalue(), f"{model}_{scenario}_{start_year}-{end_year}_all_variables.zip")

                cols = st.columns(len(var_outputs))
                for c, (band, (daily, monthly, basin_monthly)) in zip(cols, var_outputs.items()):
                    base = f"CMIP6_{model}_{band}_{scenario}_{start_year}-{end_year}"
                    c.download_button(f"{band}_basin_monthly.xlsx", to_excel_bytes(basin_monthly),
                                       f"{base}_basin_monthly.xlsx", key=f"{model}_{band}_bm")

        # ================================================================
        # 4. Maps & plots (precipitation, if extracted)
        # ================================================================
        first_model = list(model_outputs.keys())[0]
        if "pr" in model_outputs[first_model]:
            st.subheader("4. Maps & plots (precipitation)")

            first_daily, first_monthly, _ = model_outputs[first_model]["pr"]
            station_cols = [c for c in first_monthly.columns if c not in ("Year", "Month")]
            annual_per_station = first_monthly.groupby("Year")[station_cols].sum(min_count=1)
            mean_annual_station = annual_per_station.mean()

            fig_map, ax_map = plt.subplots(figsize=(6, 6))
            sc = ax_map.scatter(stations_df["Longitude"], stations_df["Latitude"],
                                 c=[mean_annual_station.get(s, np.nan) for s in stations_df["Station_ID"]],
                                 cmap="Blues", s=160, edgecolors="black")
            for _, r in stations_df.iterrows():
                ax_map.annotate(r["Station_ID"], (r["Longitude"], r["Latitude"]),
                                 xytext=(4, 4), textcoords="offset points", fontsize=8)
            cbar = fig_map.colorbar(sc, ax=ax_map)
            cbar.set_label("Mean annual rainfall (mm/yr)")
            ax_map.set_xlabel("Longitude")
            ax_map.set_ylabel("Latitude")
            ax_map.set_title(f"{first_model} — projected mean annual rainfall\n"
                              f"{scenario.upper()}, {start_year}-{end_year}")

            fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
            means = []
            for model, var_outputs in model_outputs.items():
                if "pr" not in var_outputs:
                    continue
                _, _, basin_monthly = var_outputs["pr"]
                annual = basin_monthly.groupby("Year")["Basin_Value"].sum(min_count=1)
                means.append(annual.mean())
            ax_bar.bar([m for m in model_outputs if "pr" in model_outputs[m]], means, color="#4a90d9")
            ax_bar.set_ylabel("Mean annual basin rainfall (mm/yr)")
            ax_bar.set_title(f"Projected basin-average annual rainfall\n{scenario.upper()}, {start_year}-{end_year}")
            ax_bar.tick_params(axis="x", rotation=20)
            fig_bar.tight_layout()

            fig_clim, ax_clim = plt.subplots(figsize=(8, 4.5))
            for model, var_outputs in model_outputs.items():
                if "pr" not in var_outputs:
                    continue
                _, _, basin_monthly = var_outputs["pr"]
                clim = basin_monthly.groupby("Month")["Basin_Value"].mean()
                ax_clim.plot(clim.index, clim.values, marker="o", linewidth=1.2, label=model)
            ax_clim.set_xticks(range(1, 13))
            ax_clim.set_xlabel("Month")
            ax_clim.set_ylabel("Mean monthly rainfall (mm)")
            ax_clim.set_title(f"Projected monthly climatology\n{scenario.upper()}, {start_year}-{end_year}")
            ax_clim.legend(fontsize=8)
            fig_clim.tight_layout()

            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(fig_map)
                st.download_button("Download station_rainfall_map.png", fig_to_png_bytes(fig_map), "station_rainfall_map.png")
            with p2:
                st.pyplot(fig_bar)
                st.download_button("Download annual_rainfall_by_model.png", fig_to_png_bytes(fig_bar), "annual_rainfall_by_model.png")

            st.pyplot(fig_clim)
            st.download_button("Download monthly_climatology_future.png", fig_to_png_bytes(fig_clim), "monthly_climatology_future.png")

st.markdown("---")
st.caption("App developed by Anandita Raj and Prof. Raj Mohan Singh")
