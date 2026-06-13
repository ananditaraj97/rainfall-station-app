"""
Page 7 - Documentation

Overview of the application, modules, data sources, methodology basis,
and limitations, styled as collapsible cards.
"""

import streamlit as st
from style import inject_css, sidebar_branding, footer

st.set_page_config(page_title="Documentation", layout="wide")
inject_css()
sidebar_branding()

st.title("Documentation")
st.caption("Application overview, data sources, model basis, and scope of application")

# ================================================================
# ABOUT THIS TOOL
# ================================================================
with st.container(border=True):
    st.subheader("About this tool")
    st.markdown(
        "This application provides an open-source, reproducible workflow for extracting, "
        "evaluating, and preparing climate data for hydrological climate-change impact "
        "assessment in Indian river basins. It was developed for the **Bhima River Basin** "
        "(~69,447 km², Maharashtra–Karnataka, India), and is applicable to any basin given "
        "a representative station network."
    )

    st.markdown("**Modules**")
    st.markdown(
        "- **1. IMD Station Extraction** — generates a network of basin-representative rainfall "
        "stations from the IMD 0.25° gridded daily rainfall product using a hybrid area / "
        "distance / correlation filtering and K-means clustering procedure, and computes "
        "basin-average rainfall.\n"
        "- **2. CMIP6 Models** — extracts historical (1984–2014) precipitation, temperature "
        "(max/min), relative humidity, wind speed, and solar radiation for the representative "
        "stations from the NASA NEX-GDDP-CMIP6 archive via Google Earth Engine.\n"
        "- **3. Model Evaluation & Ranking** — compares CMIP6 model precipitation against IMD "
        "observations using R², NSE, KGE, RMSE, MAE, PBIAS, and IOA, and produces a composite "
        "ranking with diagnostic plots (Taylor diagram, scatter plots, time series, climatology).\n"
        "- **4. Future Climate Projections** — extracts the same six variables under SSP2-4.5 / "
        "SSP5-8.5 scenarios for standard time slices (Near/Mid/Far future), with station maps "
        "and climatology plots.\n"
        "- **5. Ensemble & Uncertainty Analysis** — combines multiple model projections into "
        "mean/median ensembles with percentile uncertainty bands (P5–P95), fan plots, boxplots, "
        "and violin plots.\n"
        "- **6. SWAT Weather Files & Delta Factors** — converts daily climate data into "
        "SWAT-format weather files (PCP, TMP, HMD, WND, SLR) and computes monthly delta-change "
        "factors between historical and future precipitation for SWAT climate-change scenario runs."
    )

# ================================================================
# VARIABLES & UNITS
# ================================================================
with st.container(border=True):
    st.subheader("Climate variables extracted")
    st.caption("From NASA NEX-GDDP-CMIP6 (bias-corrected, statistically downscaled), via Google Earth Engine")

    var_table = {
        "Variable": ["Precipitation", "Max Temperature", "Min Temperature",
                     "Relative Humidity", "Wind Speed", "Solar Radiation"],
        "GEE band(s)": ["pr", "tasmax", "tasmin", "huss + tas (derived)", "sfcWind", "rsds"],
        "Output unit": ["mm/day", "°C", "°C", "%", "m/s", "MJ/m²/day"],
        "SWAT file": ["PCP", "TMP (max)", "TMP (min)", "HMD", "WND", "SLR"],
    }
    st.dataframe(var_table, hide_index=True, use_container_width=True)

    st.markdown(
        "**Note:** NEX-GDDP-CMIP6 does not provide a direct relative-humidity band. Relative "
        "humidity is approximated from specific humidity (`huss`) and mean air temperature "
        "(`tas`), assuming standard sea-level pressure (101,325 Pa)."
    )

# ================================================================
# DATA SOURCES
# ================================================================
with st.container(border=True):
    st.subheader("Data sources")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**IMD gridded rainfall**")
        st.caption("0.25° resolution, daily")
        st.markdown("India Meteorological Department, Pune")
    with d2:
        st.markdown("**NEX-GDDP-CMIP6**")
        st.caption("Bias-corrected, statistically downscaled CMIP6 output")
        st.markdown("Google Earth Engine catalogue: `NASA/GDDP-CMIP6`")

# ================================================================
# VALID SCOPE / DESIGN SPACE
# ================================================================
with st.container(border=True):
    st.subheader("Scope of application")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Study area**")
        st.markdown("Bhima River Basin\n\n~69,447 km²\n\nMaharashtra–Karnataka, India")
    with s2:
        st.markdown("**Historical period**")
        st.markdown("1984–2014\n\n(NEX-GDDP-CMIP6 historical experiment)")
    with s3:
        st.markdown("**Future scenarios**")
        st.markdown("SSP2-4.5, SSP5-8.5\n\nNear / Mid / Far future time slices")

# ================================================================
# NOTES & LIMITATIONS
# ================================================================
with st.container(border=True):
    st.subheader("Notes and limitations")
    st.markdown(
        "- Relative humidity is **derived**, not directly observed/modelled — see note above. "
        "This is a reasonable approximation for SWAT forcing but introduces additional "
        "uncertainty relative to a direct RH product.\n"
        "- All evaluation results are **basin-average** comparisons against a single "
        "observational reference (IMD) and do not capture within-basin spatial pattern errors.\n"
        "- Composite rankings (Tab 3) use **equal weighting** of five metrics; alternative "
        "weighting schemes could shift the ranking of closely-spaced models.\n"
        "- This tool is intended to support GCM screening and SWAT input preparation for "
        "research use. Results should be reviewed alongside the accompanying methodology "
        "before use in formal impact assessments."
    )

# ================================================================
# DISCLAIMER
# ================================================================
with st.container(border=True):
    st.subheader("Disclaimer")
    st.markdown(
        "This tool is intended for research and preliminary screening purposes only. "
        "Outputs should be verified against the source datasets and reviewed by the "
        "research team before use in publications or formal impact assessments."
    )

# ================================================================
# SOURCE CODE
# ================================================================
with st.container(border=True):
    st.subheader("Source code")
    st.markdown(
        "The complete source code for this application is available at "
        "[github.com/ananditaraj97/rainfall-station-app](https://github.com/ananditaraj97/rainfall-station-app)."
    )

footer()
