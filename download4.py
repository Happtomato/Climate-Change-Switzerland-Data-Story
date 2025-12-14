from pathlib import Path
import pandas as pd
import requests

BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
META_URL = f"{BASE_URL}/ogd-smn_meta_stations.csv"

OUTPUT_DIR = Path("data/temperature")


def get_station_codes() -> list[str]:
    """
    Download SMN station metadata and return station codes
    (lowercase, for URL construction).
    """
    print(f"Downloading SMN station metadata from {META_URL}")
    df = pd.read_csv(META_URL, sep=";", encoding="latin1")

    if "station_abbr" not in df.columns:
        raise RuntimeError("Expected column 'station_abbr' not found in SMN metadata.")

    codes = (
        df["station_abbr"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    return sorted(codes)


def download_station_yearly(code: str) -> None:
    """
    Download yearly SMN CSV for a single station.
    Example:
      ogd-smn_wae_y.csv
    """
    filename = f"ogd-smn_{code}_y.csv"
    url = f"{BASE_URL}/{code}/{filename}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  {code.upper():<4} → {url}")

    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        print(f"     FAILED (network error): {e}")
        return

    if r.status_code != 200:
        print(f"     SKIPPED (HTTP {r.status_code})")
        return

    target = OUTPUT_DIR / filename
    target.write_bytes(r.content)

    print(f"     saved → {target}")


def main():
    codes = get_station_codes()
    print(f"Found {len(codes)} SMN stations.\n")

    for code in codes:
        download_station_yearly(code)


if __name__ == "__main__":
    main()
