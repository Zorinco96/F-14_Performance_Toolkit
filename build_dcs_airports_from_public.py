# build_dcs_airports_from_public.py
import re, io, csv, math, difflib, requests
import pandas as pd
from bs4 import BeautifulSoup

MOOSE_URL = "https://flightcontrol-master.github.io/MOOSE_DOCS/Documentation/Wrapper.Airbase.html"
OA_AIRPORTS = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OA_RUNWAYS  = "https://davidmegginson.github.io/ourairports-data/runways.csv"

OUT = "dcs_airports.csv"

# Optional manual overrides: {("Map","DCS Name"):"ICAO/IATA/IDENT"}
OVERRIDE_IDENT = {
    ("Nevada","McCarran_International"): "KLAS",
    ("Nevada","Nellis"): "KLSV",
    ("Nevada","Creech"): "KINS",
    ("Nevada","North_Las_Vegas"): "KVGT",
    ("Nevada","Henderson_Executive"): "KHND",
    ("Caucasus","Batumi"): "UGSB",
    ("Caucasus","Kutaisi"): "UGKO",
    ("Caucasus","Beslan"): "URMO",  # (Mozdok is URMO; Beslan is URMO? fix manually if needed)
    ("Caucasus","Mineralnye_Vody"): "URMM",
    ("Caucasus","Nalchik"): "URMN",
    ("Caucasus","Sochi_Adler"): "URSS",
    ("Caucasus","Anapa_Vityazevo"): "URKA",
    ("Caucasus","Krasnodar_Pashkovsky"): "URKK",
    ("Caucasus","Krymsk"): "URKW",
    ("Caucasus","Maykop_Khanskaya"): "URKH",
    ("Caucasus","Tbilisi_Lochini"): "UGTB",
    ("Caucasus","Vaziani"): "UG26",  # may not exist in OA; will likely remain unmatched
    # Add more as needed
}

MAP_NAME_FIX = {
    "Caucasus":"Caucasus",
    "Nevada":"Nevada",
    "PersianGulf":"Persian Gulf",
    "Syria":"Syria",
    "Marianas":"Marianas",
    "Sinai":"Sinai",
    "SouthAtlantic":"South Atlantic",
    "Normandy":"Normandy 2.0",
    "TheChannel":"The Channel",
    "Afghanistan":"Afghanistan",
    "Kola":"Kola",  # if present
    "GermanyCW":"Germany Cold War"  # new terrain
}

def fetch_csv(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def scrape_moose(url):
    """Return list of (map, airbase_key, airbase_name_pretty)."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n")
    results = []
    # Patterns like: AIRBASE.Caucasus.Batumi
    for m in re.finditer(r"AIRBASE\.([A-Za-z0-9]+)\.([A-Za-z0-9_]+)", text):
        raw_map, raw_key = m.group(1), m.group(2)
        map_name = MAP_NAME_FIX.get(raw_map, raw_map)
        pretty = raw_key.replace("_"," ")
        results.append((map_name, raw_key, pretty))
    return results

def best_ident_for(airports_df, name_guess):
    """Find best OurAirports ident for a DCS name using fuzzy match."""
    cand_cols = ["ident","iata_code","local_code","name","municipality"]
    pool = []
    for _,r in airports_df.iterrows():
        fields = " ".join(str(r[c]) for c in cand_cols if pd.notna(r.get(c,''))).lower()
        pool.append((r["ident"], fields))
    name_low = name_guess.lower()
    scores = [(ident, difflib.SequenceMatcher(None, fields, name_low).ratio()) for ident, fields in pool]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0] if scores and scores[0][1]>0.55 else None

def tora_from_runway_end(length_ft, disp_ft):
    length_ft = float(length_ft) if pd.notna(length_ft) else float("nan")
    disp_ft   = float(disp_ft) if pd.notna(disp_ft) else 0.0
    if pd.isna(length_ft): return None
    tora = length_ft - max(0.0, disp_ft)
    return int(round(max(0.0, tora)))

def build():
    print("Downloading OurAirports...")
    airports = fetch_csv(OA_AIRPORTS)
    runways  = fetch_csv(OA_RUNWAYS)

    # Basic cleanup
    for df in (airports, runways):
        for c in df.columns:
            if df[c].dtype==object:
                df[c] = df[c].astype(str)
    if "elevation_ft" in airports.columns:
        airports["elevation_ft"] = pd.to_numeric(airports["elevation_ft"], errors="coerce")

    print("Scraping MOOSE airbase lists...")
    entries = scrape_moose(MOOSE_URL)

    rows = []
    unmatched = []

    for map_name, key, pretty in entries:
        dcs_key = (map_name, key)
        ident = OVERRIDE_IDENT.get(dcs_key)
        if not ident:
            ident = best_ident_for(airports, pretty)
        if not ident or airports[airports["ident"]==ident].empty:
            unmatched.append((map_name, pretty))
            continue

        arow = airports[airports["ident"]==ident].iloc[0]
        elev = arow.get("elevation_ft", None)
        rwy = runways[runways["airport_ident"]==ident].copy()
        if rwy.empty:
            unmatched.append((map_name, pretty+" (no runways.csv rows)"))
            continue

        # Per-end rows
        rwy["length_ft"] = pd.to_numeric(rwy["length_ft"], errors="coerce")
        for _, rr in rwy.iterrows():
            pair = f"{rr.get('le_ident','??')}/{rr.get('he_ident','??')}"
            # LE end
            le_hdg = pd.to_numeric(rr.get("le_heading_degT"), errors="coerce")
            le_len = rr.get("length_ft")
            le_disp = pd.to_numeric(rr.get("le_displaced_threshold_ft"), errors="coerce")
            tora_le = tora_from_runway_end(le_len, le_disp)

            rows.append(dict(
                map=map_name, airport_name=pretty, runway_pair=pair, runway_end=str(rr.get("le_ident","")).strip(),
                heading_deg=le_hdg if not pd.isna(le_hdg) else None, length_ft=int(le_len) if not pd.isna(le_len) else None,
                tora_ft=tora_le, toda_ft=tora_le, asda_ft=tora_le,
                threshold_elev_ft=int(elev) if not pd.isna(elev) else None,
                opp_threshold_elev_ft=None, slope_percent=None, notes="OurAirports; MOOSE-name matched"
            ))
            # HE end
            he_hdg = pd.to_numeric(rr.get("he_heading_degT"), errors="coerce")
            he_len = rr.get("length_ft")
            he_disp = pd.to_numeric(rr.get("he_displaced_threshold_ft"), errors="coerce")
            tora_he = tora_from_runway_end(he_len, he_disp)

            rows.append(dict(
                map=map_name, airport_name=pretty, runway_pair=pair, runway_end=str(rr.get("he_ident","")).strip(),
                heading_deg=he_hdg if not pd.isna(he_hdg) else None, length_ft=int(he_len) if not pd.isna(he_len) else None,
                tora_ft=tora_he, toda_ft=tora_he, asda_ft=tora_he,
                threshold_elev_ft=int(elev) if not pd.isna(elev) else None,
                opp_threshold_elev_ft=None, slope_percent=None, notes="OurAirports; MOOSE-name matched"
            ))

    out = pd.DataFrame(rows, columns=[
        "map","airport_name","runway_pair","runway_end","heading_deg","length_ft","tora_ft","toda_ft","asda_ft",
        "threshold_elev_ft","opp_threshold_elev_ft","slope_percent","notes"
    ])
    out = out.dropna(subset=["map","airport_name","runway_pair","runway_end","length_ft","tora_ft"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(out)} rows")

    if unmatched:
        with open("dcs_airports_unmatched.txt","w",encoding="utf-8") as f:
            for m,p in unmatched: f.write(f"{m},{p}\n")
        print(f"Unmatched written to dcs_airports_unmatched.txt (add to OVERRIDE_IDENT as needed).")

if __name__=="__main__":
    build()
