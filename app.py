"""
Representative Rainfall Station Generator (Tab 1 / MODEL 1)
Streamlit app implementing MODULE A - F (Hybrid method).

Run with:
    streamlit run app.py
"""

import os
import re
import glob
import calendar
import zipfile
import tempfile
import io

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
import gdown

# ============================================================
# CONSTANTS (IMD grid spec)
# ============================================================
LON_START, LAT_START = 66.5, 6.5
RES = 0.25
NCOLS, NROWS = 135, 129
CELLS_PER_DAY = NCOLS * NROWS
MISSING_VAL = -999.0


# ============================================================
# MODULE A - Grid generation + basin intersection
# ============================================================
@st.cache_data(show_spinner=False)
def generate_grid_cells():
    records = []
    half = RES / 2
    for j in range(NCOLS):
        lon = LON_START + j * RES
        for i in range(NROWS):
            lat = LAT_START + i * RES
            grid_id = f"R{i:03d}C{j:03d}"
            geom = box(lon - half, lat - half, lon + half, lat + half)
            records.append({
                "Grid_ID": grid_id, "row": i, "col": j,
                "Longitude": lon, "Latitude": lat, "geometry": geom
            })
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def load_basin_from_zip(zip_bytes):
    tmpdir = tempfile.mkdtemp()
    zpath = os.path.join(tmpdir, "basin.zip")
    with open(zpath, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmpdir)
    shp_files = glob.glob(os.path.join(tmpdir, "**", "*.shp"), recursive=True)
    if not shp_files:
        raise FileNotFoundError("No .shp file found in uploaded zip.")
    gdf = gpd.read_file(shp_files[0])
    gdf_wgs = gdf.to_crs("EPSG:4326")
    basin_union = gdf_wgs.union_all()
    return gdf_wgs, basin_union


def equal_area_crs(basin_gdf_wgs):
    cen = basin_gdf_wgs.union_all().centroid
    zone = int((cen.x + 180) / 6) + 1
    return f"EPSG:{32600 + zone}"


def find_grids_inside_basin(grid_gdf, basin_union, basin_gdf_wgs, min_area_km2):
    metric_crs = equal_area_crs(basin_gdf_wgs)
    candidates = grid_gdf[grid_gdf.intersects(basin_union)].copy()
    candidates_m = candidates.to_crs(metric_crs)
    basin_m = gpd.GeoDataFrame(geometry=[basin_union], crs="EPSG:4326").to_crs(metric_crs)
    basin_geom_m = basin_m.union_all()

    areas_km2 = [geom.intersection(basin_geom_m).area / 1e6 for geom in candidates_m.geometry]
    candidates["Represented_Area_km2"] = areas_km2

    after_area_filter = candidates[candidates["Represented_Area_km2"] >= min_area_km2].copy()
    return candidates, after_area_filter


# ============================================================
# MODULE B - Read IMD .grd files
# ============================================================
def days_in_year(year):
    return 366 if calendar.isleap(year) else 365


def extract_grd_zip(zip_bytes):
    tmpdir = tempfile.mkdtemp()
    zpath = os.path.join(tmpdir, "imd.zip")
    with open(zpath, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmpdir)
    return tmpdir


def extract_drive_file_id(drive_url):
    """Extract Google Drive file ID from common share-link formats."""
    patterns = [
        r"/d/([a-zA-Z0-9_-]{10,})",
        r"id=([a-zA-Z0-9_-]{10,})",
        r"^([a-zA-Z0-9_-]{10,})$",  # raw ID pasted directly
    ]
    for p in patterns:
        m = re.search(p, drive_url)
        if m:
            return m.group(1)
    return None


def extract_grd_zip_from_drive(drive_url):
    """Download a zip from a Google Drive share link and extract it. Returns extracted dir."""
    tmpdir = tempfile.mkdtemp()
    zpath = os.path.join(tmpdir, "imd_drive.zip")

    file_id = extract_drive_file_id(drive_url.strip())
    if not file_id:
        raise ValueError(
            "Could not parse a file ID from that link. Use the 'Anyone with the link' "
            "share link, format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
        )

    try:
        gdown.download(id=file_id, output=zpath, quiet=False)
    except Exception as e:
        raise RuntimeError(f"gdown download failed: {e}")

    if not os.path.exists(zpath) or os.path.getsize(zpath) < 1000:
        raise FileNotFoundError(
            "Download failed or file too small - check the Google Drive link is set to "
            "'Anyone with the link' (Viewer) and points directly to the zip file."
        )

    try:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmpdir)
    except zipfile.BadZipFile:
        raise ValueError(
            "Downloaded file is not a valid zip. Google Drive may have served an HTML "
            "warning page instead of the file (common for files > a few hundred MB on "
            "Streamlit Cloud). Try splitting the IMD data into smaller zips, or use the "
            "Colab notebook for the full archive."
        )
    return tmpdir


def detect_available_years(grd_dir):
    files = glob.glob(os.path.join(grd_dir, "**", "*.grd"), recursive=True)
    years = []
    for f in files:
        m = re.search(r"(\d{4})", os.path.basename(f))
        if m:
            years.append(int(m.group(1)))
    return sorted(set(years)), files


def read_grd_year(filepath, year, grid_rows, grid_cols):
    expected_days = days_in_year(year)
    data = np.fromfile(filepath, dtype=np.float32)
    actual_days = data.size // CELLS_PER_DAY
    if actual_days != expected_days:
        raise ValueError(
            f"{os.path.basename(filepath)}: expected {expected_days} days for {year}, "
            f"got {actual_days} days."
        )
    arr = data.reshape(actual_days, NROWS, NCOLS)
    series = arr[:, grid_rows, grid_cols]
    series = np.where(series == MISSING_VAL, np.nan, series)
    return series


def build_daily_matrix(grd_files_by_year, stations_df, start_year, end_year, progress_cb=None):
    grid_rows = stations_df["row"].values
    grid_cols = stations_df["col"].values
    grid_ids = stations_df["Grid_ID"].values

    all_dates, all_data = [], []
    years = range(start_year, end_year + 1)
    for idx, year in enumerate(years):
        if year not in grd_files_by_year:
            raise FileNotFoundError(f"No .grd file found for year {year}.")
        filepath = grd_files_by_year[year]
        series = read_grd_year(filepath, year, grid_rows, grid_cols)
        n_days = series.shape[0]
        dates = pd.date_range(start=f"{year}-01-01", periods=n_days, freq="D")
        all_dates.extend(dates)
        all_data.append(series)
        if progress_cb:
            progress_cb((idx + 1) / len(years), f"Read {year} ({n_days} days)")

    full_data = np.vstack(all_data)
    df = pd.DataFrame(full_data, columns=grid_ids)
    df.insert(0, "Date", all_dates)
    return df


# ============================================================
# MODULES C-F - Filters, clustering, final selection
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_filter(stations_df, thresh_km):
    df = stations_df.sort_values("Represented_Area_km2", ascending=False).reset_index(drop=True)
    kept = []
    for _, row in df.iterrows():
        too_close = any(
            haversine_km(row["Latitude"], row["Longitude"], k["Latitude"], k["Longitude"]) < thresh_km
            for k in kept
        )
        if not too_close:
            kept.append(row)
    return pd.DataFrame(kept).reset_index(drop=True)


def correlation_filter(stations_df, daily_df, thresh):
    df = stations_df.sort_values("Represented_Area_km2", ascending=False).reset_index(drop=True)
    kept_rows, kept_ids = [], []
    for _, row in df.iterrows():
        gid = row["Grid_ID"]
        too_corr = any(daily_df[gid].corr(daily_df[kid]) > thresh for kid in kept_ids)
        if not too_corr:
            kept_rows.append(row)
            kept_ids.append(gid)
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def compute_rainfall_stats(daily_df, grid_ids):
    df = daily_df.copy()
    df["Year"] = df["Date"].dt.year
    stats = {}
    for gid in grid_ids:
        series = df[gid]
        annual = df.groupby("Year")[gid].sum(min_count=1)
        stats[gid] = {
            "Mean_Annual_Rainfall_mm": annual.mean(),
            "Std_Dev_mm": series.std(),
            "CV": series.std() / series.mean() if series.mean() != 0 else np.nan,
            "Max_Daily_Rainfall_mm": series.max(),
            "Wet_Days": int((series > 1.0).sum()),
        }
    return pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(columns={"index": "Grid_ID"})


def kmeans_clustering(stats_df, min_k, max_k):
    feature_cols = ["Mean_Annual_Rainfall_mm", "Std_Dev_mm", "CV", "Max_Daily_Rainfall_mm", "Wet_Days"]
    X = stats_df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    n = len(stats_df)
    max_k = min(max_k, n - 1)
    min_k = min(min_k, max_k) if max_k >= 2 else 2
    if max_k < 2:
        stats_df = stats_df.copy()
        stats_df["Cluster_ID"] = 0
        return stats_df, 1, None, {}

    best_k, best_score, best_labels = None, -1, None
    scores = {}
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    stats_df = stats_df.copy()
    stats_df["Cluster_ID"] = best_labels
    return stats_df, best_k, best_score, scores


def select_representatives(clustered_stats_df, stations_df):
    merged = clustered_stats_df.merge(stations_df, on="Grid_ID")
    reps = []
    for cid, grp in merged.groupby("Cluster_ID"):
        rep = grp.sort_values("Represented_Area_km2", ascending=False).iloc[0]
        reps.append(rep)
    final = pd.DataFrame(reps).reset_index(drop=True)
    return final.sort_values("Cluster_ID").reset_index(drop=True)


def apply_min_area_filter(final_df, min_area):
    kept = final_df[final_df["Represented_Area_km2"] >= min_area].reset_index(drop=True)
    dropped = final_df[final_df["Represented_Area_km2"] < min_area]
    return kept, dropped


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Representative Rainfall Station Generator", layout="wide")
st.title("Representative Rainfall Station Generator")
st.caption("MODEL 1 — Tab 1: Generate representative rainfall stations from IMD 0.25° gridded data for any basin in India (Hybrid method).")

with st.sidebar:
    st.header("1. Inputs")
    basin_zip = st.file_uploader("Basin shapefile (zip containing .shp/.dbf/.shx/.prj)", type="zip")

    imd_source = st.radio("IMD .grd data source", ["Upload zip", "Google Drive link"])
    imd_zip = None
    imd_drive_url = None
    if imd_source == "Upload zip":
        imd_zip = st.file_uploader("IMD .grd files (zip, one file per year)", type="zip")
    else:
        imd_drive_url = st.text_input(
            "Google Drive shareable link to IMD .grd zip",
            help="In Drive: right-click file -> Share -> Anyone with the link -> Copy link"
        )

    st.header("2. Filter settings")
    min_grid_area = st.number_input("Minimum Grid Area (km²)", value=30, min_value=1, step=5)
    distance_thresh = st.number_input("Distance Filter (km)", value=30, min_value=1, step=5)
    corr_thresh = st.slider("Correlation Filter threshold (r)", 0.50, 0.999, 0.95, 0.01)
    min_rep_area = st.number_input("Minimum Represented Area (km²)", value=500, min_value=1, step=50)

    st.header("3. Clustering")
    cluster_mode = st.radio("Cluster count", ["Automatic (Silhouette)", "Manual"])
    if cluster_mode == "Automatic (Silhouette)":
        k_range = st.slider("Search range for optimal k", 2, 40, (10, 25))
    else:
        manual_k = st.number_input("Number of clusters", value=12, min_value=2, step=1)

imd_input_ready = (imd_zip is not None) if imd_source == "Upload zip" else bool(imd_drive_url)
run_btn = st.button("Run pipeline", type="primary", disabled=not (basin_zip and imd_input_ready))

if not (basin_zip and imd_input_ready):
    st.info("Upload basin shapefile zip and provide IMD .grd data (zip upload or Google Drive link) in the sidebar to begin.")

if run_btn:
    # ---- MODULE A ----
    with st.spinner("MODULE A: Loading basin & generating IMD grid centroids..."):
        grid_gdf = generate_grid_cells()
        basin_gdf_wgs, basin_union = load_basin_from_zip(basin_zip.getvalue())
        candidates, after_area = find_grids_inside_basin(grid_gdf, basin_union, basin_gdf_wgs, min_grid_area)

    st.subheader("Module A — Basin grid identification")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total IMD grids", f"{len(grid_gdf):,}")
    c2.metric("Grids intersecting basin", len(candidates))
    c3.metric(f"After area filter (≥{min_grid_area} km²)", len(after_area))
    st.metric("Total represented area (km²)", f"{after_area['Represented_Area_km2'].sum():,.1f}")

    # ---- MODULE B ----
    with st.spinner("MODULE B: Obtaining IMD .grd files..."):
        if imd_source == "Upload zip":
            grd_dir = extract_grd_zip(imd_zip.getvalue())
        else:
            grd_dir = extract_grd_zip_from_drive(imd_drive_url)
        available_years, grd_files = detect_available_years(grd_dir)

    if not available_years:
        st.error("No .grd files with a 4-digit year in the filename were found in the IMD zip.")
        st.stop()

    grd_files_by_year = {}
    for f in grd_files:
        m = re.search(r"(\d{4})", os.path.basename(f))
        if m:
            grd_files_by_year[int(m.group(1))] = f

    yr_min, yr_max = min(available_years), max(available_years)
    year_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max), key="year_slider")

    if st.button("Confirm year range & continue"):
        st.session_state["confirmed_years"] = year_range

    years_to_use = st.session_state.get("confirmed_years", year_range)

    progress = st.progress(0.0, text="Reading IMD .grd files...")
    daily_df = build_daily_matrix(
        grd_files_by_year, after_area, years_to_use[0], years_to_use[1],
        progress_cb=lambda frac, msg: progress.progress(frac, text=msg)
    )
    progress.empty()
    st.success(f"Daily rainfall matrix built: {daily_df.shape[0]} days x {daily_df.shape[1]-1} grids "
               f"({years_to_use[0]}-{years_to_use[1]})")

    # ---- MODULE C ----
    with st.spinner("MODULE C: Applying distance filter..."):
        after_distance = distance_filter(after_area, distance_thresh)

    # ---- MODULE D ----
    with st.spinner("MODULE D: Applying correlation filter..."):
        after_corr = correlation_filter(after_distance, daily_df, corr_thresh)

    # ---- Rainfall stats ----
    with st.spinner("Computing rainfall statistics for clustering..."):
        stats_df = compute_rainfall_stats(daily_df, after_corr["Grid_ID"].tolist())

    # ---- MODULE E ----
    with st.spinner("MODULE E: K-Means clustering..."):
        if cluster_mode == "Automatic (Silhouette)":
            clustered_stats, best_k, best_score, all_scores = kmeans_clustering(stats_df, k_range[0], k_range[1])
        else:
            clustered_stats, best_k, best_score, all_scores = kmeans_clustering(stats_df, manual_k, manual_k)

    # ---- MODULE F ----
    final = select_representatives(clustered_stats, after_corr)
    final_kept, final_dropped = apply_min_area_filter(final, min_rep_area)
    final_kept = final_kept.sort_values("Represented_Area_km2", ascending=False).reset_index(drop=True)
    final_kept.insert(0, "Station_ID", [f"ST{i+1:03d}" for i in range(len(final_kept))])

    st.subheader("Pipeline summary")
    summary = pd.DataFrame({
        "Stage": ["Total IMD grids", "Grids inside basin", "After area filter",
                  "After distance filter", "After correlation filter",
                  "After clustering", "Final representative stations"],
        "Count": [len(grid_gdf), len(candidates), len(after_area), len(after_distance),
                  len(after_corr), len(final), len(final_kept)]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

    if cluster_mode == "Automatic (Silhouette)" and best_score is not None:
        st.caption(f"Optimal k = {best_k} (Silhouette score = {best_score:.3f}). "
                   f"Scores: " + ", ".join(f"k={k}:{s:.3f}" for k, s in all_scores.items()))

    if len(final_dropped) > 0:
        st.warning(f"{len(final_dropped)} cluster representative(s) dropped "
                   f"(represented area < {min_rep_area} km²): "
                   + ", ".join(final_dropped["Grid_ID"].tolist()))

    output_cols = ["Station_ID", "Grid_ID", "Latitude", "Longitude", "Cluster_ID", "Represented_Area_km2",
                    "Mean_Annual_Rainfall_mm", "Std_Dev_mm", "CV", "Max_Daily_Rainfall_mm", "Wet_Days"]
    final_table = final_kept[output_cols]

    st.subheader(f"Final representative stations ({len(final_table)})")
    st.dataframe(final_table, hide_index=True, use_container_width=True)
    st.map(final_table.rename(columns={"Latitude": "lat", "Longitude": "lon"})[["lat", "lon"]])

    # ---- Build downstream tables ----
    rename_map = dict(zip(final_kept["Grid_ID"], final_kept["Station_ID"]))
    rep_daily = daily_df[["Date"] + list(rename_map.keys())].rename(columns=rename_map)

    rep_monthly = rep_daily.copy()
    rep_monthly["Year"] = rep_monthly["Date"].dt.year
    rep_monthly["Month"] = rep_monthly["Date"].dt.month
    monthly = rep_monthly.drop(columns="Date").groupby(["Year", "Month"]).sum(min_count=1).reset_index()

    station_cols = list(rename_map.values())
    basin_monthly = monthly[["Year", "Month"]].copy()
    basin_monthly["Basin_Rainfall_mm"] = monthly[station_cols].mean(axis=1)

    # ---- Downloads ----
    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Final_Stations.xlsx", to_excel_bytes(final_table), "Final_Stations.xlsx")
    d2.download_button("Representative_Stations_Rainfall.xlsx (daily)", to_excel_bytes(rep_daily),
                        "Representative_Stations_Rainfall.xlsx")
    d3.download_button("Representative_Stations_Monthly.xlsx", to_excel_bytes(monthly),
                        "Representative_Stations_Monthly.xlsx")
    d4.download_button("Basin_Monthly_Rainfall.xlsx", to_excel_bytes(basin_monthly),
                        "Basin_Monthly_Rainfall.xlsx")

    d5, d6 = st.columns(2)
    d5.download_button("Final_Stations.csv", final_table.to_csv(index=False).encode(), "Final_Stations.csv")
    d6.download_button("Representative_Stations_Rainfall.csv", rep_daily.to_csv(index=False).encode(),
                        "Representative_Stations_Rainfall.csv")
