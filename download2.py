"""
download_nbcn_yearly.py

Lädt für alle Swiss-NBCN-Stationen die homogenisierten Jahresdaten (y)
aus der MeteoSwiss-Collection 'ch.meteoschweiz.ogd-nbcn' herunter.

Beispiel aus der STAC-API:
  https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/sbe/ogd-nbcn_sbe_y.csv

Struktur lokal:
  data/
    meteoswiss_nbcn_meta/
      ogd-nbcn_meta_stations.csv
    meteoswiss_nbcn_yearly/
      ogd-nbcn_sbe_y.csv
      ogd-nbcn_sia_y.csv
      ...
"""

import requests
import pandas as pd
from pathlib import Path

# Basis-URL für die Collection
BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn"

# Projektstruktur
DATA_DIR = Path("data")
META_DIR = DATA_DIR / "meteoswiss_nbcn_meta"
YEARLY_DIR = DATA_DIR / "meteoswiss_nbcn_yearly"

META_DIR.mkdir(parents=True, exist_ok=True)
YEARLY_DIR.mkdir(parents=True, exist_ok=True)

meta_stations_path = META_DIR / "ogd-nbcn_meta_stations.csv"

# Falls Metadatei noch nicht da ist -> von MeteoSwiss holen
if not meta_stations_path.exists():
    print("Lade ogd-nbcn_meta_stations.csv von MeteoSwiss ...")
    resp = requests.get(
        f"{BASE_URL}/ogd-nbcn_meta_stations.csv",
        timeout=30,
    )
    resp.raise_for_status()
    meta_stations_path.write_bytes(resp.content)
    print("  -> gespeichert unter", meta_stations_path)
else:
    print("Metadatei bereits vorhanden:", meta_stations_path)

# Metadatei einlesen
stations = pd.read_csv(meta_stations_path, sep=";", encoding="cp1252")

print("Spalten in ogd-nbcn_meta_stations.csv:")
print(stations.columns.tolist())

station_abbrs = stations["station_abbr"].dropna().unique()
print(f"{len(station_abbrs)} Stationen in der Metadatei gefunden.")


def download_station_yearly(station_abbr: str):
    """
    Lädt für eine Station die Jahresdaten:
      https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/<abbr_lower>/ogd-nbcn_<abbr_lower>_y.csv
    und speichert sie im YEARLY_DIR.
    """
    abbr_lower = station_abbr.lower()
    filename = f"ogd-nbcn_{abbr_lower}_y.csv"
    url = f"{BASE_URL}/{abbr_lower}/{filename}"
    out_path = YEARLY_DIR / filename

    if out_path.exists():
        print(f"[skip] {station_abbr}: Datei existiert bereits ({out_path.name})")
        return

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            print(f"[ok]   {station_abbr}: gespeichert als {out_path.name}")
        else:
            print(f"[fail] {station_abbr}: HTTP {r.status_code} für {url}")
    except Exception as e:
        print(f"[err]  {station_abbr}: {e}")


# Alle Stationen durchgehen
for abbr in station_abbrs:
    download_station_yearly(abbr)

print("\nFertig – Jahresdaten liegen jetzt in:", YEARLY_DIR)
