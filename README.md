# DCS F-14B Takeoff Performance

A Streamlit app modeling FAA-style takeoff performance for the F-14B with DCS runway data.

## Run locally
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run f14_takeoff_app.py

## Data files
Place these CSVs either in the repo root or in ./data/:
- f14_perf.csv
- dcs_airports_expanded.csv
- intersections.csv

## Notes
- Single app entrypoint: f14_takeoff_app.py
- Core logic: f14_takeoff_core.py
- Climb helper: f14_climb_guidance.py
- Data Checker tab helps sanity-check your CSVs.
