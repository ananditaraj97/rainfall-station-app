"""
Tab 5 - Ensemble Generation & Uncertainty Analysis

Takes multiple CMIP6 basin-monthly files (from Tab 2 historical or Tab 4
future projections, same scenario/period), computes mean/median ensembles,
percentile uncertainty bands (5/25/50/75/95), and produces fan/ribbon plots,
boxplots, and ensemble-vs-individual comparison charts.
"""

import io
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

PERCENTILES = [5, 25, 50, 75, 95]


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


def model_name_from_filename(name):
    if "CMIP6_" in name:
        return name.split("CMIP6_")[-1].split("_")[0]
    return os.path.splitext(name)[0]


def load_basin_monthly_files(uploaded_files):
    """Return dict: model_name -> DataFrame (Year, Month, Basin_Rainfall_mm)."""
    models = {}
    for f in uploaded_files or []:
        df = pd.read_excel(f)
        if not {"Year", "Month", "Basin_Rainfall_mm"}.issubset(df.columns):
            st.warning(f"{f.name}: must have columns Year, Month, Basin_Rainfall_mm - skipped.")
            continue
        name = model_name_from_filename(f.name)
        models[name] = df[["Year", "Month", "Basin_Rainfall_mm"]].copy()
    return models


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Ensemble & Uncertainty Analysis", layout="wide")
st.title("Ensemble Generation & Uncertainty Analysis")
st.caption("MODEL 1 — Tab 5: Combine multiple CMIP6 model basin-monthly files (from Tab 2 historical "
           "or Tab 4 future projections, same scenario/period) into mean/median ensembles with "
           "percentile uncertainty bands (5/25/50/75/95), fan plots, and boxplots.")

st.subheader("1. Model files")
st.caption("Upload `..._basin_monthly.xlsx` files (Year, Month, Basin_Rainfall_mm) for two or more models, "
           "all from the same scenario and period (e.g. all SSP245 2021-2040, or all historical 1984-2014).")
model_files = st.file_uploader("CMIP6 basin-monthly files", type=["xlsx"], accept_multiple_files=True)

st.subheader("2. (Optional) Reference series for comparison")
ref_file = st.file_uploader("IMD Basin_Monthly_Rainfall.xlsx (historical reference, optional)", type=["xlsx"])

run_disabled = (not model_files) or (len(model_files) < 2)
if model_files and len(model_files) < 2:
    st.info("Upload at least 2 model files to build an ensemble.")

run_btn = st.button("Build ensemble & uncertainty analysis", type="primary", disabled=run_disabled)

if run_btn:
    models = load_basin_monthly_files(model_files)
    if len(models) < 2:
        st.error("Need at least 2 valid model files.")
        st.stop()

    st.success(f"Loaded {len(models)} model(s): {', '.join(models.keys())}")

    ref_df = None
    if ref_file is not None:
        raw = pd.read_excel(ref_file)
        if {"Year", "Month", "Basin_Rainfall_mm"}.issubset(raw.columns):
            ref_df = raw[["Year", "Month", "Basin_Rainfall_mm"]].copy()
        else:
            st.warning("Reference file missing Year/Month/Basin_Rainfall_mm - ignored.")

    # ---- merge all models on Year, Month ----
    merged = None
    for model, df in models.items():
        d = df.rename(columns={"Basin_Rainfall_mm": model})
        merged = d if merged is None else merged.merge(d, on=["Year", "Month"], how="inner")

    if merged is None or len(merged) == 0:
        st.error("No overlapping Year/Month rows across the uploaded model files.")
        st.stop()

    model_cols = list(models.keys())
    merged = merged.sort_values(["Year", "Month"]).reset_index(drop=True)

    # ============================================================
    # 2. Top-N ensemble subset selection
    # ============================================================
    st.subheader("3. Ensemble definition")
    st.caption("By default the ensemble uses ALL uploaded models. Optionally select a subset "
               "(e.g. your top-3 ranked models from Tab 3) for a second 'Top-N' ensemble.")
    topn_models = st.multiselect("Top-N model subset (optional)", model_cols)

    # ---- compute statistics across all models ----
    vals = merged[model_cols].values.astype(float)
    merged["Ensemble_Mean"] = vals.mean(axis=1)
    merged["Ensemble_Median"] = np.median(vals, axis=1)
    merged["Ensemble_Std"] = vals.std(axis=1)
    merged["Ensemble_CV"] = merged["Ensemble_Std"] / merged["Ensemble_Mean"].replace(0, np.nan)
    for p in PERCENTILES:
        merged[f"P{p}"] = np.percentile(vals, p, axis=1)

    if topn_models and len(topn_models) >= 1:
        topn_vals = merged[topn_models].values.astype(float)
        merged["TopN_Mean"] = topn_vals.mean(axis=1)
        merged["TopN_Median"] = np.median(topn_vals, axis=1)

    # ============================================================
    # 3. Display table
    # ============================================================
    st.subheader("4. Ensemble & uncertainty table (monthly)")
    display_cols = ["Year", "Month"] + model_cols + ["Ensemble_Mean", "Ensemble_Median", "Ensemble_Std", "Ensemble_CV"] \
        + [f"P{p}" for p in PERCENTILES]
    if "TopN_Mean" in merged.columns:
        display_cols += ["TopN_Mean", "TopN_Median"]
    st.dataframe(merged[display_cols].head(24), hide_index=True, use_container_width=True)
    st.caption(f"Showing first 24 of {len(merged)} months.")

    # ============================================================
    # 4. Annual aggregation for boxplot / annual ensemble
    # ============================================================
    annual = merged.groupby("Year")[model_cols].sum(min_count=1)
    annual_ensemble = pd.DataFrame({
        "Year": annual.index,
        "Mean": annual.mean(axis=1).values,
        "Median": annual.median(axis=1).values,
        "Std": annual.std(axis=1).values,
        "CV": (annual.std(axis=1) / annual.mean(axis=1)).values,
    })
    for p in PERCENTILES:
        annual_ensemble[f"P{p}"] = np.percentile(annual.values, p, axis=1)

    ref_annual = None
    if ref_df is not None:
        ref_annual = ref_df.groupby("Year")["Basin_Rainfall_mm"].sum(min_count=1)

    # ============================================================
    # 5. Plots
    # ============================================================
    st.subheader("5. Plots")

    merged["Date"] = pd.to_datetime(dict(year=merged["Year"], month=merged["Month"], day=1))

    # 5a. Fan / ribbon plot - monthly
    fig_fan, ax_fan = plt.subplots(figsize=(11, 4.5))
    ax_fan.fill_between(merged["Date"], merged["P5"], merged["P95"], color="#4a90d9", alpha=0.15, label="5th-95th pct")
    ax_fan.fill_between(merged["Date"], merged["P25"], merged["P75"], color="#4a90d9", alpha=0.30, label="25th-75th pct")
    ax_fan.plot(merged["Date"], merged["Ensemble_Mean"], color="#1f4e8c", linewidth=1.2, label="Ensemble mean")
    if "TopN_Mean" in merged.columns:
        ax_fan.plot(merged["Date"], merged["TopN_Mean"], color="#d62728", linewidth=1.2, label="Top-N mean")
    if ref_df is not None:
        ref_ts = ref_df.copy()
        ref_ts["Date"] = pd.to_datetime(dict(year=ref_ts["Year"], month=ref_ts["Month"], day=1))
        ref_ts = ref_ts.sort_values("Date")
        ax_fan.plot(ref_ts["Date"], ref_ts["Basin_Rainfall_mm"], color="black", linewidth=1, label="IMD reference")
    ax_fan.set_xlabel("Date")
    ax_fan.set_ylabel("Monthly basin rainfall (mm)")
    ax_fan.set_title("Ensemble monthly rainfall with uncertainty band")
    ax_fan.legend(fontsize=8, ncol=3)
    fig_fan.tight_layout()

    # 5b. Boxplot - annual totals per model
    fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
    box_data = [annual[m].values for m in model_cols]
    ax_box.boxplot(box_data, labels=model_cols, showmeans=True)
    if ref_annual is not None:
        ax_box.axhline(ref_annual.mean(), color="black", linestyle="--", linewidth=1, label="IMD mean")
        ax_box.legend(fontsize=8)
    ax_box.set_ylabel("Annual basin rainfall (mm)")
    ax_box.set_title("Distribution of annual totals by model")
    ax_box.tick_params(axis="x", rotation=30)
    fig_box.tight_layout()

    # 5c. Bar: mean annual rainfall - models, ensemble, top-N, IMD
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))
    labels = list(model_cols) + ["Ensemble Mean", "Ensemble Median"]
    values = [annual[m].mean() for m in model_cols] + [annual_ensemble["Mean"].mean(), annual_ensemble["Median"].mean()]
    colors = ["#4a90d9"] * len(model_cols) + ["#1f4e8c", "#1f4e8c"]
    if "TopN_Mean" in merged.columns:
        topn_annual_mean = merged.groupby("Year")["TopN_Mean"].sum(min_count=1).mean() \
            if False else annual[topn_models].mean(axis=1).mean()
        labels.append("Top-N Mean")
        values.append(topn_annual_mean)
        colors.append("#d62728")
    if ref_annual is not None:
        labels.append("IMD")
        values.append(ref_annual.mean())
        colors.append("#2e8b57")
    ax_bar.bar(labels, values, color=colors)
    ax_bar.set_ylabel("Mean annual basin rainfall (mm/yr)")
    ax_bar.set_title("Mean annual rainfall: individual models, ensembles, reference")
    ax_bar.tick_params(axis="x", rotation=30)
    fig_bar.tight_layout()

    # 5d. Violin plot - monthly spread per calendar month (ensemble across models)
    fig_violin, ax_violin = plt.subplots(figsize=(9, 4.5))
    monthly_groups = [merged.loc[merged["Month"] == m, model_cols].values.flatten() for m in range(1, 13)]
    ax_violin.violinplot(monthly_groups, positions=range(1, 13), showmeans=True)
    ax_violin.set_xticks(range(1, 13))
    ax_violin.set_xlabel("Month")
    ax_violin.set_ylabel("Monthly basin rainfall (mm)")
    ax_violin.set_title("Inter-model spread by calendar month")
    fig_violin.tight_layout()

    p1, p2 = st.columns(2)
    with p1:
        st.pyplot(fig_box)
        st.download_button("Download annual_boxplot.png", fig_to_png_bytes(fig_box), "annual_boxplot.png")
    with p2:
        st.pyplot(fig_bar)
        st.download_button("Download mean_annual_comparison.png", fig_to_png_bytes(fig_bar), "mean_annual_comparison.png")

    st.pyplot(fig_fan)
    st.download_button("Download ensemble_fan_plot.png", fig_to_png_bytes(fig_fan), "ensemble_fan_plot.png")

    st.pyplot(fig_violin)
    st.download_button("Download monthly_spread_violin.png", fig_to_png_bytes(fig_violin), "monthly_spread_violin.png")

    # ============================================================
    # 6. Downloads
    # ============================================================
    st.subheader("6. Downloads")
    d1, d2 = st.columns(2)
    d1.download_button("Ensemble_Basin_Monthly.xlsx", to_excel_bytes(merged.drop(columns="Date")), "Ensemble_Basin_Monthly.xlsx")
    d2.download_button("Ensemble_Annual_Uncertainty.xlsx", to_excel_bytes(annual_ensemble), "Ensemble_Annual_Uncertainty.xlsx")

st.markdown("---")
st.caption("App developed by Anandita Raj and Prof. Raj Mohan Singh")
