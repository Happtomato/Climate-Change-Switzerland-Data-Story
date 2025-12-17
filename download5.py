from pathlib import Path
import requests
import pandas as pd

# --------------------------------------------------
# BFS PxWeb tourism cube (hotel overnight stays)
# --------------------------------------------------
CUBE_ID = "px-x-1003020000_102"
PXWEB_URL = f"https://www.pxweb.bfs.admin.ch/api/v1/en/{CUBE_ID}/{CUBE_ID}.px"

OUT_DIR = Path("data") / "tourism"
OUT_PATH = OUT_DIR / "bfs_hotel_overnight_stays_cantons_monthly.csv"


# --------------------------------------------------
# JSON-stat2 → DataFrame
# --------------------------------------------------
def jsonstat2_to_df(js: dict) -> pd.DataFrame:
    dims = list(js["dimension"].keys())

    dim_vals = []
    for d in dims:
        cat = js["dimension"][d]["category"]
        if "index" in cat and isinstance(cat["index"], dict):
            keys = list(cat["index"].keys())
        elif "index" in cat:
            keys = list(cat["index"])
        else:
            keys = list(cat["label"].keys())
        dim_vals.append(keys)

    values = js["value"]
    mi = pd.MultiIndex.from_product(dim_vals, names=dims)
    df = pd.DataFrame({"value": values}, index=mi).reset_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def find_var(meta_vars: list[dict], contains: list[str]) -> dict:
    for v in meta_vars:
        txt = (v.get("text") or "").lower()
        if all(c.lower() in txt for c in contains):
            return v
    raise RuntimeError(f"Could not find variable containing {contains}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load metadata
    meta = requests.get(PXWEB_URL, timeout=60)
    meta.raise_for_status()
    meta = meta.json()
    vars_ = meta["variables"]

    var_year = find_var(vars_, ["year"])
    var_month = find_var(vars_, ["month"])
    var_canton = find_var(vars_, ["canton"])
    var_vis = find_var(vars_, ["country"])
    var_ind = find_var(vars_, ["indicator"])

    # 2) Select ALL cantons except Switzerland total (8100)
    canton_codes = [
        code for code, txt in zip(var_canton["values"], var_canton["valueTexts"])
        if code != "8100"
    ]

    visitors_total = [
        code for code, txt in zip(var_vis["values"], var_vis["valueTexts"])
        if "total" in txt.lower()
    ][0]

    overnight_code = [
        code for code, txt in zip(var_ind["values"], var_ind["valueTexts"])
        if "overnight" in txt.lower() or "logier" in txt.lower()
    ][0]

    # 3) Build query
    query = {
        "query": [
            {"code": var_year["code"], "selection": {"filter": "all", "values": ["*"]}},
            {"code": var_month["code"], "selection": {"filter": "all", "values": ["*"]}},
            {"code": var_canton["code"], "selection": {"filter": "item", "values": canton_codes}},
            {"code": var_vis["code"], "selection": {"filter": "item", "values": [visitors_total]}},
            {"code": var_ind["code"], "selection": {"filter": "item", "values": [overnight_code]}},
        ],
        "response": {"format": "json-stat2"},
    }

    r = requests.post(PXWEB_URL, json=query, timeout=180)
    r.raise_for_status()

    df = jsonstat2_to_df(r.json())

    # 4) Rename dimensions safely (PxWeb order is stable here)
    df = df.rename(columns={
        df.columns[0]: "year",
        df.columns[1]: "month",
        df.columns[2]: "Kanton",
    })

    # 5) Robust numeric conversion (FIXES 'YYYY' BUG)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["Kanton"] = pd.to_numeric(df["Kanton"], errors="coerce")

    df = df.dropna(subset=["year", "month", "Kanton", "value"]).copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["Kanton"] = df["Kanton"].astype(int)

    # 6) Save
    df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH.resolve())
    print("Rows:", len(df))
    print("Years:", df["year"].min(), "–", df["year"].max())
    print("Cantons:", df["Kanton"].nunique())
    print(df.head(10))


if __name__ == "__main__":
    main()
