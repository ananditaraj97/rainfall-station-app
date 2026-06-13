"""
Tab 3 - CMIP6 Model Evaluation & Ranking

Compares CMIP6 GCM outputs (from Tab 2, Date x Station daily, mm/day) against
IMD observed rainfall (Basin_Monthly_Rainfall.xlsx and/or
Representative_Stations_Monthly.xlsx from Tab 1), computes performance
metrics (R2, NSE, RMSE, MAE, PBIAS, KGE), ranks the models, and produces
comparison plots - all downloadable.
"""

import io
import zipfile
import tempfile
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# METRIC FUNCTIONS
# ============================================================
def r2_score_(obs, sim):
    return np.corrcoef(obs, sim)[0, 1] ** 2


def nse(obs, sim):
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim) ** 2))


def mae(obs, sim):
    return np.mean(np.abs(obs - sim))


def pbias(obs, sim):
    return 100 * np.sum(sim - obs) / np.sum(obs)


def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def compute_metrics(obs, sim):
    mask = (~np.isnan(obs)) & (~np.isnan(sim))
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2:
        return None
    return {
        "R2": r2_score_(obs, sim),
        "NSE": nse(obs, sim),
        "RMSE": rmse(obs, sim),
        "MAE": mae(obs, sim),
        "PBIAS": pbias(obs, sim),
        "KGE": kge(obs, sim),
        "n": len(obs),
    }


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


def to_monthly(df, cols):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    return d.drop(columns="Date").groupby(["Year", "Month"])[cols].sum(min_count=1).reset_index()


def build_monthly_and_basin(daily_df):
    """From a Date x Station daily DataFrame, build monthly (Year, Month, stations)
    and basin-average monthly (Year, Month, Basin_Rainfall_mm)."""
    station_cols = [c for c in daily_df.columns if c != "Date"]
    monthly = to_monthly(daily_df, station_cols)
    basin_monthly = monthly[["Year", "Month"]].copy()
    basin_monthly["Basin_Rainfall_mm"] = monthly[station_cols].mean(axis=1)
    return monthly, basin_monthly


def classify_file(name, df):
    """Classify a single uploaded/extracted Excel file as IMD or CMIP6 model data,
    and return its monthly-station and basin-monthly DataFrames (whichever derivable).
    Returns None if the file isn't a recognised rainfall data format (e.g. Final_Stations.xlsx)."""
    cols = set(df.columns)
    name_lower = name.lower()
    is_model = "cmip6" in name_lower

    model_name = None
    if is_model:
        if "CMIP6_" in name:
            model_name = name.split("CMIP6_")[-1].split("_")[0]
        else:
            model_name = os.path.splitext(name)[0]

    monthly_station = None
    basin_monthly = None

    if "Date" in cols:
        station_cols = [c for c in df.columns if c != "Date"]
        if not station_cols:
            return None
        monthly_station, basin_monthly = build_monthly_and_basin(df)
    elif {"Year", "Month", "Basin_Rainfall_mm"}.issubset(cols):
        basin_monthly = df[["Year", "Month", "Basin_Rainfall_mm"]].copy()
    elif {"Year", "Month"}.issubset(cols) and len(cols) > 2:
        station_cols = [c for c in df.columns if c not in ("Year", "Month")]
        monthly_station = df[["Year", "Month"] + station_cols].copy()
        basin_monthly = monthly_station[["Year", "Month"]].copy()
        basin_monthly["Basin_Rainfall_mm"] = monthly_station[station_cols].mean(axis=1)
    else:
        return None  # unrecognised format (e.g. Final_Stations.xlsx)

    return {"is_model": is_model, "model_name": model_name,
            "monthly_station": monthly_station, "basin_monthly": basin_monthly}


def load_and_classify_all(uploaded_files, zip_file):
    """Read all uploaded files (individual + zip), classify each as IMD or model data,
    and aggregate into: imd_basin_monthly, imd_station_monthly,
    model_monthly_basin (dict), model_monthly_station (dict)."""
    imd_basin_monthly = None
    imd_station_monthly = None
    model_monthly_basin = {}
    model_monthly_station = {}
    skipped = []

    def handle(name, df):
        nonlocal imd_basin_monthly, imd_station_monthly
        rec = classify_file(name, df)
        if rec is None:
            skipped.append(name)
            return
        if rec["is_model"]:
            m = rec["model_name"]
            if rec["monthly_station"] is not None:
                model_monthly_station[m] = rec["monthly_station"]
                model_monthly_basin[m] = rec["basin_monthly"]  # derived from station, preferred
            elif rec["basin_monthly"] is not None and m not in model_monthly_basin:
                model_monthly_basin[m] = rec["basin_monthly"]
        else:
            if rec["monthly_station"] is not None:
                imd_station_monthly = rec["monthly_station"]
                imd_basin_monthly = rec["basin_monthly"]  # derived from station, preferred
            elif rec["basin_monthly"] is not None and imd_basin_monthly is None:
                imd_basin_monthly = rec["basin_monthly"]

    for f in uploaded_files or []:
        df = pd.read_excel(f)
        handle(f.name, df)

    if zip_file is not None:
        tmpdir = tempfile.mkdtemp()
        zpath = os.path.join(tmpdir, "data.zip")
        with open(zpath, "wb") as fh:
            fh.write(zip_file.getvalue())
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmpdir)
        for fpath in sorted(glob.glob(os.path.join(tmpdir, "**", "*.xlsx"), recursive=True)):
            name = os.path.basename(fpath)
            df = pd.read_excel(fpath)
            handle(name, df)

    return imd_basin_monthly, imd_station_monthly, model_monthly_basin, model_monthly_station, skipped


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="CMIP6 Model Evaluation & Ranking", layout="wide")
st.title("CMIP6 Model Evaluation & Ranking")
st.caption("MODEL 1 — Tab 3: Compare NEX-GDDP-CMIP6 bias-corrected, downscaled precipitation against "
           "IMD observed rainfall (R2, NSE, RMSE, MAE, PBIAS, KGE), rank models, and view comparison plots.")

st.subheader("1. Data — IMD + CMIP6 models")
st.caption(
    "Easiest: put all your files in one zip — IMD outputs from Tab 1 "
    "(Representative_Stations_Rainfall / _Monthly, Basin_Monthly_Rainfall — daily, monthly, "
    "or basin-monthly, any of these work) plus any number of NEX-GDDP-CMIP6 (bias-corrected, downscaled) "
    "model files from Tab 2 "
    "(`CMIP6_<model>_..._daily/monthly/basin_monthly.xlsx`). "
    "The app auto-detects which file is IMD and which are models (by filename/columns), "
    "and computes both basin-average and station-wise R2/NSE/etc wherever data allows."
)
combined_zip = st.file_uploader("Combined zip (IMD + CMIP6 model files)", type=["zip"])

with st.expander("Or upload files individually (no zip)"):
    individual_files = st.file_uploader(
        "IMD + CMIP6 Excel files (any of: Representative_Stations_Rainfall/_Monthly, "
        "Basin_Monthly_Rainfall, CMIP6_<model>_*.xlsx)",
        type=["xlsx"], accept_multiple_files=True
    )

run_btn = st.button("Run evaluation", type="primary",
                     disabled=(combined_zip is None and not individual_files))

if combined_zip is None and not individual_files:
    st.info("Upload a combined zip (or individual files) containing IMD data and one or more CMIP6 model files to begin.")

if run_btn:
    imd_basin_monthly, imd_station_monthly, model_monthly_basin, model_monthly_station, skipped = \
        load_and_classify_all(individual_files, combined_zip)

    if skipped:
        st.caption(f"Skipped (unrecognised format): {', '.join(skipped)}")

    if imd_basin_monthly is None:
        st.error("No IMD reference data found (need Basin_Monthly_Rainfall, "
                 "Representative_Stations_Monthly, or Representative_Stations_Rainfall).")
        st.stop()

    if not model_monthly_basin:
        st.error("No CMIP6 model files found (filenames must contain 'CMIP6').")
        st.stop()

    st.success(f"IMD reference loaded. Found {len(model_monthly_basin)} model(s): "
               f"{', '.join(model_monthly_basin.keys())}"
               + (" (station-wise data also available)" if imd_station_monthly is not None else
                  " (basin-average only — no IMD station-wise file found)"))
    # BASIN-AVERAGE METRICS
    # ============================================================
    st.subheader("2. Basin-average metrics & ranking")
    basin_results = []
    for model, mdf in model_monthly_basin.items():
        merged = imd_basin_monthly.merge(mdf, on=["Year", "Month"], suffixes=("_obs", "_sim"))
        obs = merged["Basin_Rainfall_mm_obs"].values.astype(float)
        sim = merged["Basin_Rainfall_mm_sim"].values.astype(float)
        m = compute_metrics(obs, sim)
        if m is None:
            st.warning(f"{model}: insufficient overlapping months with IMD data, skipped.")
            continue
        m["Model"] = model
        basin_results.append(m)

    if not basin_results:
        st.error("No models had overlapping data with the IMD basin record.")
        st.stop()

    basin_metrics_df = pd.DataFrame(basin_results)[["Model", "R2", "NSE", "RMSE", "MAE", "PBIAS", "KGE", "n"]]
    st.dataframe(basin_metrics_df, hide_index=True, use_container_width=True)

    ranking = basin_metrics_df.copy()
    ranking["Rank_NSE"] = ranking["NSE"].rank(ascending=False)
    ranking["Rank_KGE"] = ranking["KGE"].rank(ascending=False)
    ranking["Rank_RMSE"] = ranking["RMSE"].rank(ascending=True)
    ranking["Overall_Rank_Score"] = ranking[["Rank_NSE", "Rank_KGE", "Rank_RMSE"]].mean(axis=1)
    ranking = ranking.sort_values("Overall_Rank_Score").reset_index(drop=True)

    st.markdown("**Model ranking** (lower Overall_Rank_Score = better)")
    st.dataframe(ranking, hide_index=True, use_container_width=True)

    # ============================================================
    # STATION-WISE METRICS (optional)
    # ============================================================
    station_metrics_df = None
    if imd_station_monthly is not None:
        st.subheader("2b. Station-wise metrics")
        sta_results = []
        imd_station_cols = [c for c in imd_station_monthly.columns if c not in ("Year", "Month")]
        for model, mdf in model_monthly_station.items():
            common_cols = [c for c in imd_station_cols if c in mdf.columns]
            if not common_cols:
                continue
            merged = imd_station_monthly.merge(mdf, on=["Year", "Month"], suffixes=("_obs", "_sim"))
            for st_col in common_cols:
                obs = merged[f"{st_col}_obs"].values.astype(float)
                sim = merged[f"{st_col}_sim"].values.astype(float)
                m = compute_metrics(obs, sim)
                if m is None:
                    continue
                m["Model"] = model
                m["Station"] = st_col
                sta_results.append(m)

        if sta_results:
            station_metrics_df = pd.DataFrame(sta_results)[["Model", "Station", "R2", "NSE", "RMSE", "MAE", "PBIAS", "KGE", "n"]]
            st.dataframe(station_metrics_df, hide_index=True, use_container_width=True)
        else:
            st.info("No matching station columns between IMD and model files - station-wise metrics skipped.")

    # ============================================================
    # PLOTS
    # ============================================================
    st.subheader("3. Plots")
    figs = {}

    # 4a. Scatter plots (basin monthly, IMD vs each model)
    n_models = len(model_monthly_basin)
    fig_scatter, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    for ax, (model, mdf) in zip(axes[0], model_monthly_basin.items()):
        merged = imd_basin_monthly.merge(mdf, on=["Year", "Month"], suffixes=("_obs", "_sim"))
        x = merged["Basin_Rainfall_mm_obs"].values.astype(float)
        y = merged["Basin_Rainfall_mm_sim"].values.astype(float)
        ax.scatter(x, y, alpha=0.5, s=15)
        lims = [0, max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="1:1")
        r2 = r2_score_(x, y)
        ax.set_title(f"{model} (R2={r2:.2f})")
        ax.set_xlabel("IMD monthly rainfall (mm)")
        ax.set_ylabel("Model monthly rainfall (mm)")
        ax.legend()
    fig_scatter.tight_layout()
    figs["scatter_plots"] = fig_scatter

    # 4b. Ranking bar chart (NSE, KGE, R2 grouped)
    fig_rank, ax_rank = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ranking))
    width = 0.25
    ax_rank.bar(x - width, ranking["NSE"], width, label="NSE")
    ax_rank.bar(x, ranking["KGE"], width, label="KGE")
    ax_rank.bar(x + width, ranking["R2"], width, label="R2")
    ax_rank.set_xticks(x)
    ax_rank.set_xticklabels(ranking["Model"], rotation=30)
    ax_rank.set_ylabel("Score")
    ax_rank.set_title("Model ranking (NSE, KGE, R2)")
    ax_rank.legend()
    fig_rank.tight_layout()
    figs["ranking_chart"] = fig_rank

    # 4c. Basin monthly time series: IMD vs all models
    imd_ts = imd_basin_monthly.copy()
    imd_ts["Date"] = pd.to_datetime(dict(year=imd_ts["Year"], month=imd_ts["Month"], day=1))
    imd_ts = imd_ts.sort_values("Date")

    fig_ts, ax_ts = plt.subplots(figsize=(11, 4.5))
    ax_ts.plot(imd_ts["Date"], imd_ts["Basin_Rainfall_mm"], color="black", linewidth=1.2, label="IMD")
    for model, mdf in model_monthly_basin.items():
        m_ts = mdf.copy()
        m_ts["Date"] = pd.to_datetime(dict(year=m_ts["Year"], month=m_ts["Month"], day=1))
        m_ts = m_ts.sort_values("Date")
        ax_ts.plot(m_ts["Date"], m_ts["Basin_Rainfall_mm"], linewidth=0.8, alpha=0.8, label=model)
    ax_ts.set_xlabel("Year")
    ax_ts.set_ylabel("Monthly basin rainfall (mm)")
    ax_ts.set_title("Basin-average monthly rainfall: IMD vs CMIP6 models")
    ax_ts.legend(ncol=min(len(model_monthly_basin) + 1, 6), fontsize=8)
    fig_ts.tight_layout()
    figs["timeseries_comparison"] = fig_ts

    # 4d. Monthly climatology comparison
    imd_clim = imd_basin_monthly.groupby("Month")["Basin_Rainfall_mm"].mean()
    fig_clim, ax_clim = plt.subplots(figsize=(8, 4.5))
    ax_clim.plot(imd_clim.index, imd_clim.values, color="black", linewidth=2, marker="o", label="IMD")
    for model, mdf in model_monthly_basin.items():
        m_clim = mdf.groupby("Month")["Basin_Rainfall_mm"].mean()
        ax_clim.plot(m_clim.index, m_clim.values, linewidth=1, marker="o", alpha=0.8, label=model)
    ax_clim.set_xticks(range(1, 13))
    ax_clim.set_xlabel("Month")
    ax_clim.set_ylabel("Mean monthly rainfall (mm)")
    ax_clim.set_title("Monthly climatology: IMD vs CMIP6 models")
    ax_clim.legend(fontsize=8)
    fig_clim.tight_layout()
    figs["climatology_comparison"] = fig_clim

    # 4e. Taylor-diagram-style plot
    fig_taylor, ax_t = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    obs_vals = imd_basin_monthly["Basin_Rainfall_mm"].values.astype(float)
    for model, mdf in model_monthly_basin.items():
        merged = imd_basin_monthly.merge(mdf, on=["Year", "Month"], suffixes=("_obs", "_sim"))
        x = merged["Basin_Rainfall_mm_obs"].values.astype(float)
        y = merged["Basin_Rainfall_mm_sim"].values.astype(float)
        r = np.corrcoef(x, y)[0, 1]
        std_ratio = y.std() / x.std()
        theta = np.arccos(np.clip(r, -1, 1))
        ax_t.plot(theta, std_ratio, "o", markersize=8, label=model)
    ax_t.plot(0, 1, "k*", markersize=15, label="IMD (reference)")
    ax_t.set_thetamin(0)
    ax_t.set_thetamax(90)
    ax_t.set_title("Taylor diagram (correlation vs std-dev ratio)")
    ax_t.legend(bbox_to_anchor=(1.3, 1.0), fontsize=8)
    figs["taylor_diagram"] = fig_taylor

    # ---- display ----
    p1, p2 = st.columns(2)
    with p1:
        st.pyplot(fig_scatter)
        st.download_button("Download scatter_plots.png", fig_to_png_bytes(fig_scatter), "scatter_plots.png")
    with p2:
        st.pyplot(fig_rank)
        st.download_button("Download ranking_chart.png", fig_to_png_bytes(fig_rank), "ranking_chart.png")

    st.pyplot(fig_ts)
    st.download_button("Download timeseries_comparison.png", fig_to_png_bytes(fig_ts), "timeseries_comparison.png")

    p3, p4 = st.columns(2)
    with p3:
        st.pyplot(fig_clim)
        st.download_button("Download climatology_comparison.png", fig_to_png_bytes(fig_clim), "climatology_comparison.png")
    with p4:
        st.pyplot(fig_taylor)
        st.download_button("Download taylor_diagram.png", fig_to_png_bytes(fig_taylor), "taylor_diagram.png")

    # ============================================================
    # DOWNLOADS
    # ============================================================
    st.subheader("4. Downloads")
    d1, d2 = st.columns(2)
    d1.download_button("Model_Metrics_Basin.xlsx", to_excel_bytes(basin_metrics_df), "Model_Metrics_Basin.xlsx")
    d2.download_button("Model_Ranking.xlsx", to_excel_bytes(ranking), "Model_Ranking.xlsx")
    if station_metrics_df is not None:
        st.download_button("Model_Metrics_PerStation.xlsx", to_excel_bytes(station_metrics_df), "Model_Metrics_PerStation.xlsx")

st.markdown("---")
st.caption("App developed by Anandita Raj and Prof. Raj Mohan Singh")
