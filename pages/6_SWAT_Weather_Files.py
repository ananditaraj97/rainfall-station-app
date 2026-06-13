"""
Tab 6 - SWAT Weather File Generator & Climate Change Delta Factors

Part A: converts a daily Date x Station precipitation file (from Tab 2
historical or Tab 4 future projections) into SWAT-format .pcp text files
(one per station), packaged as a downloadable zip.

Part B: computes monthly climate-change delta factors (% change in
precipitation by calendar month) between a historical baseline and a
future projection - the standard "delta-change" input for SWAT scenario
runs.

SWAT PCP file format (per station):
  line 1: start date as YYYYMMDD
  line 2..: one daily precipitation value (mm), 2 decimal places
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


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


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="SWAT Weather Files & Delta Factors", layout="wide")
st.title("SWAT Weather File Generator & Climate Change Delta Factors")
st.caption("MODEL 1 — Tab 6: Convert daily CMIP6/IMD precipitation into SWAT-format .pcp files, "
           "and compute monthly delta-change factors (% change) between historical and future "
           "precipitation for SWAT climate-change scenario runs.")

# ================================================================
# PART A: SWAT PCP FILE GENERATOR
# ================================================================
st.header("A. SWAT precipitation (.pcp) file generator")
st.caption("Upload a daily Date x Station file (e.g. Representative_Stations_Rainfall.xlsx from "
           "Tab 1, or any `..._daily.xlsx` from Tab 2/Tab 4). One .pcp file is generated per station.")

daily_file = st.file_uploader("Daily precipitation file (Date + station columns)", type=["xlsx"], key="pcp_upload")

if daily_file is not None:
    df = pd.read_excel(daily_file, parse_dates=["Date"])
    if "Date" not in df.columns:
        st.error("File must contain a 'Date' column.")
        st.stop()

    station_cols = [c for c in df.columns if c != "Date"]
    df = df.sort_values("Date").reset_index(drop=True)

    # check for missing dates (SWAT requires a continuous daily series)
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing_dates = full_range.difference(df["Date"])
    if len(missing_dates) > 0:
        st.warning(f"{len(missing_dates)} missing date(s) in the series - these will be filled with 0.00 "
                   f"to keep the SWAT file continuous (first missing: {missing_dates[0].date()}).")
        df = df.set_index("Date").reindex(full_range).reset_index().rename(columns={"index": "Date"})

    start_date_str = df["Date"].iloc[0].strftime("%Y%m%d")
    n_days = len(df)
    st.info(f"{len(station_cols)} station(s), {n_days} days, start date {start_date_str}")

    # filename mapping: pcp1.txt, pcp2.txt, ... in column order
    mapping = pd.DataFrame({
        "Station_ID": station_cols,
        "PCP_File": [f"pcp{i+1}.txt" for i in range(len(station_cols))],
    })
    st.dataframe(mapping, hide_index=True)

    if st.button("Generate SWAT .pcp files (zip)", type="primary"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, station in enumerate(station_cols):
                lines = [start_date_str]
                values = df[station].fillna(0).values
                for v in values:
                    lines.append(f"{v:.2f}")
                content = "\n".join(lines) + "\n"
                zf.writestr(f"pcp{i+1}.txt", content)
            # also include the station -> filename mapping
            mapping_buf = io.BytesIO()
            with pd.ExcelWriter(mapping_buf, engine="openpyxl") as writer:
                mapping.to_excel(writer, index=False)
            zf.writestr("pcp_station_mapping.xlsx", mapping_buf.getvalue())

        buf.seek(0)
        st.download_button("Download SWAT_PCP_files.zip", buf.getvalue(), "SWAT_PCP_files.zip")
        st.success(f"Generated {len(station_cols)} .pcp file(s), {n_days} days each.")

st.markdown("---")

# ================================================================
# PART B: CLIMATE CHANGE DELTA FACTORS
# ================================================================
st.header("B. Climate change delta factors (% change by month)")
st.caption("Upload a historical baseline (e.g. Basin_Monthly_Rainfall.xlsx from Tab 1, or a "
           "historical CMIP6 `..._basin_monthly.xlsx` from Tab 2) and a future projection "
           "(`..._basin_monthly.xlsx` from Tab 4). Delta factor for each calendar month = "
           "100 x (Future_mean - Historical_mean) / Historical_mean.")

c1, c2 = st.columns(2)
hist_file = st.file_uploader("Historical basin-monthly file (Year, Month, Basin_Rainfall_mm)", type=["xlsx"], key="hist_delta")
fut_file = st.file_uploader("Future basin-monthly file (Year, Month, Basin_Rainfall_mm)", type=["xlsx"], key="fut_delta")

run_delta = st.button("Compute delta factors", type="primary", disabled=(hist_file is None or fut_file is None))

if run_delta:
    hist_df = pd.read_excel(hist_file)
    fut_df = pd.read_excel(fut_file)

    for name, d in [("Historical", hist_df), ("Future", fut_df)]:
        if not {"Year", "Month", "Basin_Rainfall_mm"}.issubset(d.columns):
            st.error(f"{name} file must have columns Year, Month, Basin_Rainfall_mm. Found: {list(d.columns)}")
            st.stop()

    hist_clim = hist_df.groupby("Month")["Basin_Rainfall_mm"].mean()
    fut_clim = fut_df.groupby("Month")["Basin_Rainfall_mm"].mean()

    delta_table = pd.DataFrame({
        "Month": range(1, 13),
        "Month_Name": MONTH_NAMES,
        "Historical_Mean_mm": [hist_clim.get(m, np.nan) for m in range(1, 13)],
        "Future_Mean_mm": [fut_clim.get(m, np.nan) for m in range(1, 13)],
    })
    delta_table["Delta_Percent"] = 100 * (delta_table["Future_Mean_mm"] - delta_table["Historical_Mean_mm"]) / delta_table["Historical_Mean_mm"]
    delta_table["Delta_Factor"] = 1 + delta_table["Delta_Percent"] / 100  # multiplier: Future = Historical x Delta_Factor

    st.subheader("Delta factors table")
    st.dataframe(delta_table, hide_index=True, use_container_width=True)

    annual_hist = hist_df.groupby("Year")["Basin_Rainfall_mm"].sum(min_count=1).mean()
    annual_fut = fut_df.groupby("Year")["Basin_Rainfall_mm"].sum(min_count=1).mean()
    annual_delta = 100 * (annual_fut - annual_hist) / annual_hist
    st.metric("Annual mean rainfall change", f"{annual_delta:+.1f}%",
              help=f"Historical: {annual_hist:.1f} mm/yr, Future: {annual_fut:.1f} mm/yr")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d62728" if v < 0 else "#2e8b57" for v in delta_table["Delta_Percent"]]
    ax.bar(delta_table["Month_Name"], delta_table["Delta_Percent"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Change in mean monthly rainfall (%)")
    ax.set_title("Climate change delta factors (Future vs Historical)")
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("Download delta_factors_chart.png", fig_to_png_bytes(fig), "delta_factors_chart.png")

    st.download_button("Download Delta_Factors.xlsx", to_excel_bytes(delta_table), "Delta_Factors.xlsx")

    st.caption(
        "To apply: multiply each day's historical precipitation in the corresponding calendar month "
        "by Delta_Factor to obtain a delta-perturbed future daily series for SWAT (delta-change method). "
        "For quantile-mapping bias correction instead of simple delta-change, use the historical "
        "IMD vs historical CMIP6 series from Tabs 1-2 to derive correction functions, then apply "
        "them to the future series before generating .pcp files in Part A."
    )

st.markdown("---")
st.caption("App developed by Anandita Raj and Prof. Raj Mohan Singh")
