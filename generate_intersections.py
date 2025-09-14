import pandas as pd
from pathlib import Path

# Percentages of TORA to create for each runway end
PCTS = [0.95, 0.85, 0.70, 0.50]  # 95%, 85%, 70%, 50%

def main():
    src = Path("dcs_airports_expanded.csv")
    if not src.exists():
        raise SystemExit("Missing dcs_airports_expanded.csv in repo root.")
    df = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        tora = float(r["tora_ft"])
        toda = float(r["toda_ft"])
        asda = float(r["asda_ft"])

        # Label ends consistently: A* for the first seen end, B* for the paired opposite
        # We'll just use A* for every row (per-end basis) because we store rows per runway_end.
        # That keeps names unique under (map, airport, pair, end, name).
        base = {
            "map": r["map"], "airport_name": r["airport_name"],
            "runway_pair": r["runway_pair"], "runway_end": r["runway_end"]
        }
        for i, p in enumerate(PCTS, start=1):
            name = f"A{i}"
            rows.append({
                **base,
                "name": name,
                "tora_ft": round(tora * p),
                "toda_ft": round(toda * p),
                "asda_ft": round(asda * p),
                "notes": f"Est ~{int(p*100)}% — verify"
            })

    out = pd.DataFrame(rows, columns=["map","airport_name","runway_pair","runway_end","name","tora_ft","toda_ft","asda_ft","notes"])
    out.to_csv("intersections.csv", index=False)
    print(f"intersections.csv written with {len(out)} rows across {df.shape[0]} runway ends.")

if __name__ == "__main__":
    main()
