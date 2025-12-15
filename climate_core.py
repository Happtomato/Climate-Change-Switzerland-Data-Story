# =========================
# climate_core.py (ENGLISH)
# =========================
from pathlib import Path
import struct

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
DATA_DIR = Path("data")
SWISS_MEAN_PATH = DATA_DIR / "climate-data-swissmean_regSwiss_1.4.txt"

METEOSWISS_META_DIR = DATA_DIR / "meteoswiss_nbcn_meta"
METEOSWISS_YEARLY_DIR = DATA_DIR / "meteoswiss_nbcn_yearly"

GLACIER_BASE_DIR = DATA_DIR / "glaciers"

SNOW_DIR = DATA_DIR / "snow"
NIME_META_STATIONS_URL = (
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/ogd-nime_meta_stations.csv"
)

SMN_TEMP_DIR = DATA_DIR / "temperature"
PARAM_TMAX_ABS = "tre200yx"
SMN_META_PATH = DATA_DIR / "temperature" / "ogd-smn_meta_stations.csv"

PARAM_TEMP_YEAR = "ths200y0"  # homogenized annual mean temperature


# -------------------------------------------------------------------
# Swiss mean temperature
# -------------------------------------------------------------------
def load_swiss_mean() -> pd.DataFrame:
    ch = pd.read_csv(
        SWISS_MEAN_PATH,
        sep="\t",
        skiprows=15,
    )
    ch = ch[["time", "year"]].rename(columns={"time": "year_int", "year": "temp_mean"})
    ch["year_int"] = ch["year_int"].astype(int)
    ch = ch.dropna(subset=["temp_mean"]).reset_index(drop=True)
    return ch


def compute_ch_anomalies(ch: pd.DataFrame):
    # Baseline period 1961–1990
    baseline_mask = (ch["year_int"] >= 1961) & (ch["year_int"] <= 1990)
    baseline_mean = ch.loc[baseline_mask, "temp_mean"].mean()

    ch = ch.copy()
    ch["anomaly"] = ch["temp_mean"] - baseline_mean
    ch["anomaly_smooth"] = (
        ch["anomaly"].rolling(window=11, center=True, min_periods=5).mean()
    )

    # Warming since start: first 30 years vs. last 10 years
    early_mask = ch["year_int"].between(ch["year_int"].min(), ch["year_int"].min() + 29)
    recent_mask = ch["year_int"].between(ch["year_int"].max() - 9, ch["year_int"].max())

    warming_since_start = (
        ch.loc[recent_mask, "temp_mean"].mean()
        - ch.loc[early_mask, "temp_mean"].mean()
    )

    latest_year = int(ch["year_int"].max())
    return ch, warming_since_start, latest_year


def make_fig1(ch: pd.DataFrame, warming_since_start: float, latest_year: int):
    extreme_years = [2003, 2015, 2022]
    ext_df = ch[ch["year_int"].isin(extreme_years)]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ch["year_int"],
            y=ch["anomaly"],
            mode="lines",
            name="Annual values",
            line=dict(width=1.5),
            hovertemplate="Year %{x}<br>Anomaly %{y:.2f} °C<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ch["year_int"],
            y=ch["anomaly_smooth"],
            mode="lines",
            name="11-year mean",
            line=dict(width=4),
            hovertemplate="Year %{x}<br>Trend %{y:.2f} °C<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ext_df["year_int"],
            y=ext_df["anomaly"],
            mode="markers+text",
            name="Notable hot years",
            marker=dict(size=9, symbol="circle"),
            text=[str(y) for y in ext_df["year_int"]],
            textposition="top center",
            hovertemplate="Year %{x}<br>Anomaly %{y:.2f} °C<extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            "How much has Switzerland warmed?<br>"
            "<span style='font-size:0.8em'>"
            f"Since measurements began ≈ +{warming_since_start:.1f} °C "
            f"(baseline 1961–1990, latest full year {latest_year})"
            "</span>"
        ),
        xaxis_title="Year",
        yaxis_title="Temperature anomaly (°C, baseline 1961–1990)",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# -------------------------------------------------------------------
# Station trends
# -------------------------------------------------------------------
def load_station_series(path: Path, parameter: str = PARAM_TEMP_YEAR):
    df = pd.read_csv(path, sep=";", encoding="cp1252")

    if parameter not in df.columns:
        return None

    if "reference_timestamp" in df.columns:
        t = pd.to_datetime(df["reference_timestamp"], errors="coerce")
        year = t.dt.year
    elif "year" in df.columns:
        year = df["year"]
    else:
        return None

    out = pd.DataFrame({"year": year, "value": df[parameter]})
    out = out.dropna(subset=["year", "value"])
    if out.empty:
        return None

    out["year"] = out["year"].astype(int)
    return out


def compute_trend_df():
    csv_files = sorted(METEOSWISS_YEARLY_DIR.glob("ogd-nbcn_*_y.csv"))
    trend_rows = []

    for csv_path in csv_files:
        try:
            station_abbr = csv_path.stem.split("_")[1].upper()
        except Exception:
            continue

        df_station = load_station_series(csv_path)
        if df_station is None or df_station.empty:
            continue

        mask = (df_station["year"] >= 1961) & (df_station["year"] <= 2024)
        df_period = df_station.loc[mask].dropna(subset=["value"])

        if df_period["year"].nunique() < 20:
            continue

        x = df_period["year"].values.astype(float)
        y = df_period["value"].values.astype(float)

        slope, _ = np.polyfit(x, y, 1)
        trend_per_decade = slope * 10.0  # °C per decade

        trend_rows.append(
            {
                "station_abbr": station_abbr,
                "trend_degC_per_decade": trend_per_decade,
                "n_years": df_period["year"].nunique(),
            }
        )

    return pd.DataFrame(trend_rows)


def enrich_trend_with_meta(trend_df: pd.DataFrame):
    stations = pd.read_csv(
        METEOSWISS_META_DIR / "ogd-nbcn_meta_stations.csv",
        sep=";",
        encoding="cp1252",
    )

    station_cols = [
        "station_abbr",
        "station_name",
        "station_canton",
        "station_height_masl",
        "station_coordinates_wgs84_lat",
        "station_coordinates_wgs84_lon",
    ]

    trend_df = trend_df.merge(stations[station_cols], on="station_abbr", how="left")

    # Simple region classification (for narrative grouping only)
    def classify_region(row):
        h = row["station_height_masl"]
        canton = row["station_canton"]
        if h >= 1000:
            return "Alpine region"
        elif canton in ["TI", "VS", "GR"]:
            return "South"
        else:
            return "North"

    trend_df["region"] = trend_df.apply(classify_region, axis=1)
    return trend_df


def make_fig2(trend_df: pd.DataFrame):
    vmin = trend_df["trend_degC_per_decade"].min()
    vmax = trend_df["trend_degC_per_decade"].max()

    center_lat = trend_df["station_coordinates_wgs84_lat"].mean()
    center_lon = trend_df["station_coordinates_wgs84_lon"].mean()

    trend_df = trend_df.copy()
    trend_df["hover"] = (
        "<b>" + trend_df["station_name"] + "</b><br>"
        + "Canton: " + trend_df["station_canton"] + "<br>"
        + "Region: " + trend_df["region"] + "<br>"
        + "Elevation: " + trend_df["station_height_masl"].astype(int).astype(str) + " m a.s.l.<br>"
        + "Trend: " + trend_df["trend_degC_per_decade"].round(2).astype(str) + " °C / decade<br>"
        + "Lat/Lon: "
        + trend_df["station_coordinates_wgs84_lat"].round(3).astype(str) + "°, "
        + trend_df["station_coordinates_wgs84_lon"].round(3).astype(str) + "°"
    )

    fig = px.scatter_map(
        trend_df,
        lat="station_coordinates_wgs84_lat",
        lon="station_coordinates_wgs84_lon",
        color="trend_degC_per_decade",
        size=np.abs(trend_df["trend_degC_per_decade"]) * 12,
        color_continuous_scale="Turbo",
        range_color=(min(vmin, 0), max(vmax, 0.7)),
        zoom=6,
        hover_name=None,
        hover_data={"hover": True},
        title="Warming per decade since 1961 at Swiss NBCN stations",
        height=650,
    )

    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        map={"center": {"lat": center_lat, "lon": center_lon}},
        coloraxis_colorbar=dict(title="Trend<br>°C / decade"),
    )

    return fig


# -------------------------------------------------------------------
# Glaciers
# -------------------------------------------------------------------
def _read_shp_polygon_areas(shp_path: Path) -> float:
    """
    Reads a .shp file containing Polygon geometries and computes the total area
    using the shoelace formula.

    Note: This is a lightweight reader for simple polygon shapefiles.
    """
    with open(shp_path, "rb") as f:
        f.read(100)  # skip 100-byte shapefile header
        total_area = 0.0

        while True:
            header = f.read(8)
            if not header:
                break

            _, content_length_words = struct.unpack(">2i", header)
            content_length_bytes = content_length_words * 2
            content = f.read(content_length_bytes)
            if not content:
                break

            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type != 5:  # 5 = Polygon
                continue

            num_parts = struct.unpack("<i", content[36:40])[0]
            num_points = struct.unpack("<i", content[40:44])[0]

            parts_offset = 44
            parts = []
            for i in range(num_parts):
                part_index = struct.unpack(
                    "<i", content[parts_offset + 4 * i : parts_offset + 4 * (i + 1)]
                )[0]
                parts.append(part_index)
            parts.append(num_points)

            points_offset = 44 + 4 * num_parts
            points = []
            for i in range(num_points):
                x, y = struct.unpack(
                    "<2d", content[points_offset + 16 * i : points_offset + 16 * (i + 1)]
                )
                points.append((x, y))

            for p in range(num_parts):
                start = parts[p]
                end = parts[p + 1]
                pts = points[start:end]

                xs = np.array([pt[0] for pt in pts])
                ys = np.array([pt[1] for pt in pts])

                area = 0.5 * abs(
                    np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
                )
                total_area += area

        # shapefile coordinates are in meters → convert to km²
        return total_area / 1_000_000


def load_glacier_inventory_totals(base_dir: Path = GLACIER_BASE_DIR) -> pd.DataFrame:
    rows = []
    for shp in sorted(base_dir.rglob("SGI_*.shp")):
        year = int(shp.stem.split("_")[1])  # SGI_1931 → 1931
        total_area_km2 = _read_shp_polygon_areas(shp)
        rows.append({"year": year, "total_area_km2": total_area_km2})

    if not rows:
        return pd.DataFrame(columns=["year", "total_area_km2"])

    return pd.DataFrame(rows).sort_values("year")


def make_fig_glaciers(glaciers_df: pd.DataFrame):
    if glaciers_df.empty:
        return None

    fig = px.line(
        glaciers_df,
        x="year",
        y="total_area_km2",
        markers=True,
        labels={"year": "Inventory year", "total_area_km2": "Total glacier area [km²]"},
        title="Shrinking glacier area in Switzerland",
    )

    fig.update_traces(line=dict(width=3), fill="tozeroy")
    fig.update_yaxes(range=[0, glaciers_df["total_area_km2"].max() * 1.05])
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


# -------------------------------------------------------------------
# Snow days from NIME station files (hns000d0 > 0)
# -------------------------------------------------------------------
def load_nime_station_meta() -> pd.DataFrame:
    """
    Loads NIME station metadata and returns:
      station, height_masl, region_class (Lowlands/Alps)
    """
    try:
        meta = pd.read_csv(NIME_META_STATIONS_URL, sep=";", encoding="latin1")
    except Exception:
        return pd.DataFrame(columns=["station", "height_masl", "region_class"])

    cols_lower = {c.lower(): c for c in meta.columns}

    if "nat_abbr" in cols_lower:
        station_col = cols_lower["nat_abbr"]
    elif "station_abbr" in cols_lower:
        station_col = cols_lower["station_abbr"]
    else:
        station_col = meta.columns[0]

    height_candidates = [
        c for c in meta.columns if any(k in c.lower() for k in ["height", "hoehe", "altitude"])
    ]
    if not height_candidates:
        return pd.DataFrame(columns=["station", "height_masl", "region_class"])
    height_col = height_candidates[0]

    out = pd.DataFrame(
        {
            "station": meta[station_col].astype(str).str.upper().str.strip(),
            "height_masl": pd.to_numeric(meta[height_col], errors="coerce"),
        }
    ).dropna(subset=["height_masl"])

    def classify(h):
        return "Alps" if h >= 1000 else "Lowlands"

    out["region_class"] = out["height_masl"].apply(classify)
    return out


def load_snow_days_from_nime(base_dir: Path = SNOW_DIR) -> pd.DataFrame:
    """
    Reads all ogd-nime_*_d_*.csv files and computes:

    - For each station and year: number of days with fresh snow height > 0 cm
      (column 'hns000d0', missing values are ignored).
    - Joins with station elevation from NIME metadata.
    - Splits into:
        * Lowlands  (< 1000 m a.s.l.)
        * Alps      (>= 1000 m a.s.l.)
    - Keeps long-term stations per region (>= 80% coverage of years)
      and averages snow days per year.

    Returns DataFrame with:
        year,
        mean_snow_days_lowlands,
        mean_snow_days_alps,
        n_stations_lowlands,
        n_stations_alps
    """
    if not base_dir.exists():
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_lowlands",
                "mean_snow_days_alps",
                "n_stations_lowlands",
                "n_stations_alps",
            ]
        )

    files = sorted(base_dir.glob("ogd-nime_*_d_*.csv"))
    if not files:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_lowlands",
                "mean_snow_days_alps",
                "n_stations_lowlands",
                "n_stations_alps",
            ]
        )

    meta = load_nime_station_meta()
    if meta.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_lowlands",
                "mean_snow_days_alps",
                "n_stations_lowlands",
                "n_stations_alps",
            ]
        )

    per_station_year = []

    for path in files:
        try:
            df = pd.read_csv(path, sep=";", encoding="latin1")
        except Exception:
            continue

        df.columns = [c.lower() for c in df.columns]
        if "reference_timestamp" not in df.columns or "hns000d0" not in df.columns:
            continue

        if "station_abbr" in df.columns:
            station = df["station_abbr"].astype(str).str.upper().iloc[0]
        else:
            parts = path.stem.split("_")
            station = parts[2].upper() if len(parts) >= 3 else "UNK"

        df["reference_timestamp"] = pd.to_datetime(
            df["reference_timestamp"], dayfirst=True, errors="coerce"
        )
        df = df.dropna(subset=["reference_timestamp"])
        df["year"] = df["reference_timestamp"].dt.year

        fresh = pd.to_numeric(df["hns000d0"], errors="coerce")
        df["snow_day"] = fresh >= 1.0  # keep your >0cm logic (>=1.0 matches your existing code)

        df = df.dropna(subset=["snow_day"])

        per_year = df.groupby("year")["snow_day"].sum().reset_index()
        per_year["station"] = station
        per_station_year.append(per_year)

    if not per_station_year:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_lowlands",
                "mean_snow_days_alps",
                "n_stations_lowlands",
                "n_stations_alps",
            ]
        )

    all_sy = pd.concat(per_station_year, ignore_index=True)

    # limit to years from 1966 onwards (as in your original code)
    all_sy = all_sy[all_sy["year"] >= 1966]
    if all_sy.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_lowlands",
                "mean_snow_days_alps",
                "n_stations_lowlands",
                "n_stations_alps",
            ]
        )

    all_sy = all_sy.merge(meta, on="station", how="left").dropna(
        subset=["height_masl", "region_class"]
    )

    start_year = 1966
    end_year = int(all_sy["year"].max())
    full_years = end_year - start_year + 1

    summaries = []

    for region_name, region_label in [("Lowlands", "lowlands"), ("Alps", "alps")]:
        reg = all_sy[all_sy["region_class"] == region_name].copy()
        if reg.empty:
            continue

        cov = reg.groupby("station")["year"].nunique().reset_index(name="n_years")
        cov["coverage"] = cov["n_years"] / full_years
        long_term = cov.loc[cov["coverage"] >= 0.8, "station"]

        reg_long = reg[reg["station"].isin(long_term)]
        if reg_long.empty:
            reg_long = reg

        reg_summary = reg_long.groupby("year")["snow_day"].agg(["mean", "count"]).reset_index()
        reg_summary = reg_summary.rename(
            columns={
                "mean": f"mean_snow_days_{region_label}",
                "count": f"n_stations_{region_label}",
            }
        )
        summaries.append(reg_summary)

    if not summaries:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_lowlands",
                "mean_snow_days_alps",
                "n_stations_lowlands",
                "n_stations_alps",
            ]
        )

    out = summaries[0]
    for extra in summaries[1:]:
        out = out.merge(extra, on="year", how="outer")

    out = out.sort_values("year").reset_index(drop=True)
    return out


def make_fig_snow_days(snow_df: pd.DataFrame):
    """
    Two lines: mean snow days per year,
    split by Lowlands vs. Alps.
    """
    if snow_df.empty:
        return None

    plot_df = snow_df.copy()

    rename_map = {}
    if "mean_snow_days_lowlands" in plot_df.columns:
        rename_map["mean_snow_days_lowlands"] = "Lowlands"
    if "mean_snow_days_alps" in plot_df.columns:
        rename_map["mean_snow_days_alps"] = "Alps"

    plot_df = plot_df.rename(columns=rename_map)

    y_cols = [c for c in ["Lowlands", "Alps"] if c in plot_df.columns]
    if not y_cols:
        return None

    fig = px.line(
        plot_df,
        x="year",
        y=y_cols,
        markers=True,
        labels={
            "year": "Year",
            "value": "Average snow days per station",
            "variable": "Region",
        },
        title="Snow days are declining Lowlands vs. Alps",
    )

    fig.update_traces(line=dict(width=3))
    fig.update_layout(template="plotly_white", hovermode="x unified")

    max_val = 0
    for col in y_cols:
        max_val = max(max_val, float(plot_df[col].max()))
    fig.update_yaxes(range=[0, max_val * 1.05])

    return fig


# -------------------------------------------------------------------
# Annual absolute maximum temperature (SMN)
# -------------------------------------------------------------------
def load_smn_absolute_max_temperature(base_dir: Path = SMN_TEMP_DIR) -> pd.DataFrame:
    """
    Loads yearly SMN station files and extracts annual absolute maximum temperature
    (tre200yx).

    Returns mean per year for:
      - Lowlands (< 1000 m a.s.l.)
      - Alps (>= 1000 m a.s.l.)
    """
    files = sorted(base_dir.glob("ogd-smn_*_y.csv"))
    print(f"SMN files found: {len(files)}")
    if not files:
        return pd.DataFrame()

    # Load SMN metadata (prefer local cache if available)
    if SMN_META_PATH.exists():
        meta = pd.read_csv(SMN_META_PATH, sep=";", encoding="latin1")
    else:
        meta = pd.read_csv(
            "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv",
            sep=";",
            encoding="latin1",
        )
        SMN_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(SMN_META_PATH, sep=";", index=False, encoding="latin1")

    meta = meta[["station_abbr", "station_height_masl"]].dropna()
    meta["station_abbr"] = meta["station_abbr"].str.upper()

    rows = []

    for path in files:
        try:
            df = pd.read_csv(path, sep=";", encoding="latin1")
        except Exception:
            continue

        if "tre200yx" not in df.columns:
            continue

        if "year" in df.columns:
            year = df["year"]
        elif "year_end" in df.columns:
            year = df["year_end"]
        elif "year_start" in df.columns:
            year = df["year_start"]
        elif "time" in df.columns:
            year = pd.to_datetime(df["time"], errors="coerce").dt.year
        elif "reference_timestamp" in df.columns:
            year = pd.to_datetime(df["reference_timestamp"], errors="coerce").dt.year
        else:
            continue

        sub = pd.DataFrame({"year": year, "tmax_abs": df["tre200yx"]}).dropna()
        if sub.empty:
            continue

        station = path.stem.split("_")[1].upper()
        sub["station_abbr"] = station
        rows.append(sub)

    print(f"Rows before merge: {len(rows)}")
    if not rows:
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True)

    all_df = all_df.merge(meta, on="station_abbr", how="left")
    all_df = all_df.dropna(subset=["station_height_masl"])

    all_df["region"] = np.where(all_df["station_height_masl"] >= 1000, "Alps", "Lowlands")

    out = all_df.groupby(["year", "region"])["tmax_abs"].mean().reset_index()

    print("Final SMN rows:", len(out))
    print(out.head())
    return out


def make_fig_absolute_max_temperature(temp_df: pd.DataFrame):
    if temp_df.empty:
        return None

    fig = px.line(
        temp_df,
        x="year",
        y="tmax_abs",
        color="region",
        markers=True,
        labels={
            "year": "Year",
            "tmax_abs": "Annual absolute maximum temperature [°C]",
            "region": "Region",
        },
        title="Extreme heat is becoming more common",
    )

    fig.update_traces(line=dict(width=3))
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


# -------------------------------------------------------------------
# Convenience: prepare everything
# -------------------------------------------------------------------
def prepare_all():
    # Swiss mean
    ch = load_swiss_mean()
    ch, warming_since_start, latest_year = compute_ch_anomalies(ch)
    fig1 = make_fig1(ch, warming_since_start, latest_year)

    # Station trends
    trend_df = compute_trend_df()
    trend_df = enrich_trend_with_meta(trend_df)
    fig2 = make_fig2(trend_df)

    # Glaciers
    glaciers_df = load_glacier_inventory_totals()
    fig3 = make_fig_glaciers(glaciers_df) if not glaciers_df.empty else None

    # Snow
    snow_df = load_snow_days_from_nime()
    fig4 = make_fig_snow_days(snow_df)

    # Annual absolute max temperature
    temp_max_df = load_smn_absolute_max_temperature()
    fig6 = make_fig_absolute_max_temperature(temp_max_df)

    return (
        ch,
        fig1,
        trend_df,
        fig2,
        warming_since_start,
        latest_year,
        glaciers_df,
        fig3,
        snow_df,
        fig4,
        temp_max_df,
        fig6,
    )