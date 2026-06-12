# Representative Rainfall Station Generator + CMIP6 Comparison

Generates representative rainfall stations from IMD 0.25° gridded daily
rainfall data for any basin in India (Hybrid method: area filter -> distance
filter -> correlation filter -> K-Means/Silhouette clustering), and lets users
download matching CMIP6 (NEX-GDDP-CMIP6) GCM precipitation for R2/NSE comparison.

## Repo file structure (upload all of these)

```
app.py
requirements.txt
README.md
.gitignore
pages/
  2_CMIP6_Models.py
Representative_Rainfall_Station_Generator.ipynb
CMIP6_GEE_Extraction.ipynb
CMIP6_Model_Evaluation_Ranking.ipynb
```

## Tab 1 - Representative Rainfall Station Generator (app.py)

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

### Inputs
1. Basin shapefile - zip containing .shp .dbf .shx .prj
2. IMD .grd files - zip with one file per year (filename must contain the 4-digit year),
   OR a Google Drive shareable link (set sharing to "Anyone with the link")

### Outputs
- Final_Stations - Station_ID, lat/lon, cluster, represented area, rainfall stats
- Representative_Stations_Rainfall - daily rainfall matrix (Date x stations)
- Representative_Stations_Monthly - monthly aggregated rainfall per station
- Basin_Monthly_Rainfall - basin-average monthly rainfall
- Plots (PNG download): station map, mean annual rainfall, monthly climatology, basin monthly time series

### Default filter settings (adjustable in app)

| Filter | Default |
|---|---|
| Minimum grid area | 30 km2 |
| Distance filter | 30 km |
| Correlation filter | r > 0.95 |
| Minimum represented area | 500 km2 |
| Cluster count (k) | Auto, search range 10-25 |

## Tab 2 - CMIP6 GCM Extraction (pages/2_CMIP6_Models.py)

Streamlit multi-page app - this file appears automatically as a second page
in the sidebar nav once uploaded under pages/.

### Setup (one-time)

1. Create a GEE-enabled Google Cloud project + service account, download its JSON key
2. Register the service account email at https://signup.earthengine.google.com/#!/service_accounts
3. In Streamlit Cloud: app -> Settings -> Secrets -> add the JSON fields under
   [gee_service_account] (see service account JSON for field names/values)

### Usage
1. Upload Final_Stations.xlsx (from Tab 1) for station coordinates
2. Pick up to 5 GCM models (out of 34 available), year range (default 1984-2014, historical scenario)
3. Click "Extract CMIP6 data" - converts kg/m2/s -> mm/day, same Date x Station format as IMD
4. Download one Excel per model. Run additional batches of <=5 models as needed.

## Deploy as a public website (free)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io -> sign in with GitHub
3. "New app" -> select this repo -> main file = app.py -> Deploy
4. Public URL: https://<your-app-name>.streamlit.app

## Run in Google Colab (alternative, no install/deploy needed)

- Representative_Rainfall_Station_Generator.ipynb - Tab 1 pipeline
- CMIP6_GEE_Extraction.ipynb - extract CMIP6 data (all 34 models, no 5-model limit)
- CMIP6_Model_Evaluation_Ranking.ipynb - R2/NSE/RMSE/PBIAS/KGE metrics, ranking, Taylor diagram

Open each in Colab and run cells top to bottom.
