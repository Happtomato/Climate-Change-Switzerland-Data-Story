import requests
import os

BASE_COLLECTION = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-nbcn"
ASSETS_URL      = BASE_COLLECTION + "/assets"

out_dir = "data/meteoswiss_nbcn_meta"
os.makedirs(out_dir, exist_ok=True)

r = requests.get(ASSETS_URL)
r.raise_for_status()
assets = r.json()["assets"]

for asset in assets:
    href = asset["href"]
    fname = asset["id"]  # z.B. ogd-nbcn_meta_stations.csv
    print("Lade:", fname, "von", href)
    resp = requests.get(href)
    resp.raise_for_status()
    with open(os.path.join(out_dir, fname), "wb") as f:
        f.write(resp.content)

print("Meta-Dateien gespeichert in:", out_dir)
