from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime"
META_STATIONS_URL = f"{BASE_URL}/ogd-nime_meta_stations.csv"

# Where to save all station files
OUTPUT_DIR = Path("data/snow")


def get_station_codes(meta_url: str = META_STATIONS_URL) -> list[str]:
    """
    Download ogd-nime_meta_stations.csv and return the list of
    3-letter station codes (e.g. ['mst', 'ber', 'lug', ...]) in lower case.
    """
    print(f"Downloading station metadata from {meta_url} ...")
    df = pd.read_csv(meta_url, sep=";", encoding="latin1")

    # MeteoSwiss docs say: station abbreviation column is 'nat_abbr'
    if "nat_abbr" in df.columns:
        codes = (
            df["nat_abbr"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist()
        )
    else:
        # Fallback: assume first column holds the station codes
        first_col = df.columns[0]
        print(f"'nat_abbr' not found, falling back to first column: {first_col}")
        codes = (
            df[first_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist()
        )

    # use lower-case codes in URLs as per naming convention
    return [c.lower() for c in codes]


def download_station_historical(code: str) -> None:
    """
    Download the daily historical CSV for a single station
    and save it under data/snow/.
    Example URL:
      .../mst/ogd-nime_mst_d_historical.csv
    """
    filename = f"ogd-nime_{code}_d_historical.csv"
    url = f"{BASE_URL}/{code}/{filename}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  {code.upper()}: {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
    except Exception as e:
        print(f"    -> FAILED (network error): {e}")
        return

    if response.status_code != 200:
        print(f"    -> SKIPPED (HTTP {response.status_code})")
        return

    target_path = OUTPUT_DIR / filename
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"    -> saved to {target_path}")


def main():
    codes = get_station_codes()
    print(f"Found {len(codes)} stations in metadata.")

    for code in codes:
        download_station_historical(code)


if __name__ == "__main__":
    main()
