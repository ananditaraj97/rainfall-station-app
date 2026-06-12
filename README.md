# Representative Rainfall Station Generator

Generates representative rainfall stations from IMD 0.25° gridded daily
rainfall data for any basin in India (Hybrid method: area filter → distance
filter → correlation filter → K-Means/Silhouette clustering).

## Files

- `app.py` — Streamlit web app (full pipeline with sliders, map, downloads)
- `Representative_Rainfall_Station_Generator.ipynb` — Google Colab notebook (same pipeline, no install needed)
- `requirements.txt` — Python dependencies

## Run the web app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

### Inputs
1. Basin shapefile — zip containing `.shp .dbf .shx .prj`
2. IMD `.grd` files — zip with one file per year (filename must contain the 4-digit year),
   OR a Google Drive shareable link (set sharing to "Anyone with the link")

## Deploy as a public website (free)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io → sign in with GitHub
3. "New app" → select this repo → main file = `app.py` → Deploy
4. Public URL: `https://<your-app-name>.streamlit.app`

## Run in Google Colab

Open `Representative_Rainfall_Station_Generator.ipynb` in Colab and run cells top to bottom.

## Outputs

- `Final_Stations` — Station_ID, lat/lon, cluster, represented area, rainfall stats
- `Representative_Stations_Rainfall` — daily rainfall matrix (Date × stations)
- `Representative_Stations_Monthly` — monthly aggregated rainfall per station
- `Basin_Monthly_Rainfall` — basin-average monthly rainfall

## Default filter settings (adjustable in app)

| Filter | Default |
|---|---|
| Minimum grid area | 30 km² |
| Distance filter | 30 km |
| Correlation filter | r > 0.95 |
| Minimum represented area | 500 km² |
| Cluster count (k) | Auto, search range 10-25 |
