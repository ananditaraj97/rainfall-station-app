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
from style import inject_css, sidebar_branding, footer


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
inject_css()
sidebar_branding()
st.title("SWAT Weather File Generator & Climate Change Delta Factors")
st.caption("Tab 6: Convert daily CMIP6/IMD precipitation into SWAT-format .pcp files, "
           "and compute monthly delta-change factors (% change) between historical and future "
           "precipitation for SWAT climate-change scenario runs.")

# ================================================================
# PART A: SWAT WEATHER FILE GENERATOR
# ================================================================
st.header("A. SWAT weather file generator")
st.caption("Generate SWAT-format weather files (PCP, TMP, HMD, WND, SLR) from CMIP6/IMD daily data. "
           "Two options: a single precipitation file (PCP only), or the full "
           "`..._all_variables.zip` from Tab 2/Tab 4 (all 5 SWAT files at once).")

SWAT_VAR_MAP = {
    "pr": "pcp", "rh": "hmd", "sfcWind": "wnd", "rsds": "slr",
}

a_mode = st.radio("Input type", ["Single precipitation file (PCP only)", "all_variables.zip (full weather set)"])


def align_daily(df):
    """Sort, fill missing dates with 0, return df + start date string + station columns."""
    station_cols = [c for c in df.columns if c != "Date"]
    df = df.sort_values("Date").reset_index(drop=True)
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing = full_range.difference(df["Date"])
    if len(missing) > 0:
        df = df.set_index("Date").reindex(full_range).reset_index().rename(columns={"index": "Date"})
    start_date_str = df["Date"].iloc[0].strftime("%Y%m%d")
    return df, start_date_str, station_cols


def single_value_files(df, station_cols, start_date_str, prefix):
    """One value per line, 2dp. Returns dict filename -> text content."""
    files = {}
    for i, station in enumerate(station_cols):
        lines = [start_date_str] + [f"{v:.2f}" for v in df[station].fillna(0).values]
        files[f"{prefix}{i+1}.txt"] = "\n".join(lines) + "\n"
    return files


def tmp_files(tmax_df, tmin_df, station_cols, start_date_str):
    """TMAX,TMIN comma-separated pairs per line."""
    files = {}
    for i, station in enumerate(station_cols):
        tmax_vals = tmax_df[station].fillna(0).values
        tmin_vals = tmin_df[station].fillna(0).values
        lines = [start_date_str] + [f"{a:.2f},{b:.2f}" for a, b in zip(tmax_vals, tmin_vals)]
        files[f"tmp{i+1}.txt"] = "\n".join(lines) + "\n"
    return files


if a_mode == "Single precipitation file (PCP only)":
    st.caption("Upload a daily Date x Station precipitation file (e.g. "
               "Representative_Stations_Rainfall.xlsx from Tab 1, or `..._pr_..._daily.xlsx` "
               "from Tab 2/Tab 4).")
    daily_file = st.file_uploader("Daily precipitation file (Date + station columns)", type=["xlsx"], key="pcp_upload")

    if daily_file is not None:
        df = pd.read_excel(daily_file, parse_dates=["Date"])
        if "Date" not in df.columns:
            st.error("File must contain a 'Date' column.")
            st.stop()

        df, start_date_str, station_cols = align_daily(df)
        st.info(f"{len(station_cols)} station(s), {len(df)} days, start date {start_date_str}")

        mapping = pd.DataFrame({
            "Station_ID": station_cols,
            "PCP_File": [f"pcp{i+1}.txt" for i in range(len(station_cols))],
        })
        st.dataframe(mapping, hide_index=True)

        if st.button("Generate SWAT .pcp files (zip)", type="primary"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, content in single_value_files(df, station_cols, start_date_str, "pcp").items():
                    zf.writestr(fname, content)
                mapping_buf = io.BytesIO()
                with pd.ExcelWriter(mapping_buf, engine="openpyxl") as writer:
                    mapping.to_excel(writer, index=False)
                zf.writestr("pcp_station_mapping.xlsx", mapping_buf.getvalue())
            buf.seek(0)
            st.download_button("Download SWAT_PCP_files.zip", buf.getvalue(), "SWAT_PCP_files.zip")
            st.success(f"Generated {len(station_cols)} .pcp file(s), {len(df)} days each.")

else:
    st.caption("Upload the `<model>_..._all_variables.zip` downloaded from Tab 2 (historical) or "
               "Tab 4 (future) for ONE model. This contains daily files for all extracted "
               "variables (pr, tasmax, tasmin, rh, sfcWind, rsds).")
    var_zip = st.file_uploader("all_variables.zip", type=["zip"], key="swat_zip_upload")

    if var_zip is not None:
        import re as _re
        tmpdir_buf = io.BytesIO(var_zip.getvalue())
        zf_in = zipfile.ZipFile(tmpdir_buf)

        # find daily files per variable: CMIP6_<model>_<var>_..._daily.xlsx
        var_daily = {}
        for name in zf_in.namelist():
            if not name.endswith("_daily.xlsx"):
                continue
            m = _re.match(r"CMIP6_.+?_(pr|tasmax|tasmin|rh|sfcWind|rsds)_.*_daily\.xlsx", name)
            if not m:
                continue
            band = m.group(1)
            var_daily[band] = pd.read_excel(io.BytesIO(zf_in.read(name)), parse_dates=["Date"])

        if not var_daily:
            st.error("No recognised `<var>_..._daily.xlsx` files found in the zip.")
        else:
            found = list(var_daily.keys())
            st.success(f"Found daily data for: {', '.join(found)}")

            # use station columns from the first available variable as the canonical order
            ref_band = found[0]
            ref_df, start_date_str, station_cols = align_daily(var_daily[ref_band])

            mapping_rows = []
            files = {}

            if "pr" in var_daily:
                df_pr, sds, scols = align_daily(var_daily["pr"])
                files.update(single_value_files(df_pr, scols, sds, "pcp"))
                for i, s in enumerate(scols):
                    mapping_rows.append({"Station_ID": s, "SWAT_File": f"pcp{i+1}.txt", "Variable": "Precipitation (PCP)"})

            if "tasmax" in var_daily and "tasmin" in var_daily:
                df_tmax, sds, scols = align_daily(var_daily["tasmax"])
                df_tmin, _, _ = align_daily(var_daily["tasmin"])
                files.update(tmp_files(df_tmax, df_tmin, scols, sds))
                for i, s in enumerate(scols):
                    mapping_rows.append({"Station_ID": s, "SWAT_File": f"tmp{i+1}.txt", "Variable": "Temperature TMAX,TMIN (TMP)"})
            elif "tasmax" in var_daily or "tasmin" in var_daily:
                st.warning("Both tasmax AND tasmin are needed for the SWAT .tmp file - only one was found, skipping TMP.")

            for band, swat_prefix in [("rh", "hmd"), ("sfcWind", "wnd"), ("rsds", "slr")]:
                if band in var_daily:
                    df_v, sds, scols = align_daily(var_daily[band])
                    files.update(single_value_files(df_v, scols, sds, swat_prefix))
                    label = {"rh": "Relative Humidity (HMD)", "sfcWind": "Wind Speed (WND)", "rsds": "Solar Radiation (SLR)"}[band]
                    for i, s in enumerate(scols):
                        mapping_rows.append({"Station_ID": s, "SWAT_File": f"{swat_prefix}{i+1}.txt", "Variable": label})

            mapping = pd.DataFrame(mapping_rows)
            st.dataframe(mapping, hide_index=True, use_container_width=True)
            st.info(f"{len(station_cols)} station(s), {len(ref_df)} days, start date {start_date_str}, "
                    f"{len(files)} SWAT file(s) will be generated.")

            if st.button("Generate SWAT weather files (zip)", type="primary"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, content in files.items():
                        zf.writestr(fname, content)
                    mapping_buf = io.BytesIO()
                    with pd.ExcelWriter(mapping_buf, engine="openpyxl") as writer:
                        mapping.to_excel(writer, index=False)
                    zf.writestr("swat_station_mapping.xlsx", mapping_buf.getvalue())
                buf.seek(0)
                st.download_button("Download SWAT_weather_files.zip", buf.getvalue(), "SWAT_weather_files.zip")
                st.success(f"Generated {len(files)} SWAT weather file(s).")

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

footer()
