from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import struct


# -------------------------------------------------------------------
# Pfade & Konstanten
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

PARAM_TEMP_YEAR = "ths200y0"  # homogenisierte Jahresmitteltemperatur


# -------------------------------------------------------------------
# Schweizer Mitteltemperatur
# -------------------------------------------------------------------
def load_swiss_mean():
    ch = pd.read_csv(
        SWISS_MEAN_PATH,
        sep="\t",
        skiprows=15,
    )
    ch = ch[["time", "year"]].rename(
        columns={"time": "year_int", "year": "temp_mean"}
    )
    ch["year_int"] = ch["year_int"].astype(int)
    ch = ch.dropna(subset=["temp_mean"]).reset_index(drop=True)
    return ch


def compute_ch_anomalies(ch: pd.DataFrame):
    # Referenzperiode 1961–1990
    baseline_mask = (ch["year_int"] >= 1961) & (ch["year_int"] <= 1990)
    baseline_mean = ch.loc[baseline_mask, "temp_mean"].mean()

    ch = ch.copy()
    ch["anomaly"] = ch["temp_mean"] - baseline_mean
    ch["anomaly_smooth"] = (
        ch["anomaly"]
        .rolling(window=11, center=True, min_periods=5)
        .mean()
    )

    # Seit Messbeginn: erste 30 Jahre vs. letzte 10 Jahre
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
            name="Jahreswerte",
            line=dict(width=1.5),
            hovertemplate="Jahr %{x}<br>Anomalie %{y:.2f} °C<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ch["year_int"],
            y=ch["anomaly_smooth"],
            mode="lines",
            name="11-Jahres-Mittel",
            line=dict(width=4),
            hovertemplate="Jahr %{x}<br>Trend %{y:.2f} °C<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ext_df["year_int"],
            y=ext_df["anomaly"],
            mode="markers+text",
            name="Extremjahre",
            marker=dict(size=9, symbol="circle"),
            text=[str(y) for y in ext_df["year_int"]],
            textposition="top center",
            hovertemplate="Jahr %{x}<br>Anomalie %{y:.2f} °C<extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            f"Wie stark hat sich die Schweiz erwärmt?<br>"
            f"<span style='font-size:0.8em'>"
            f"Seit Messbeginn ≈ +{warming_since_start:.1f} °C "
            f"(Referenz 1961–1990, letzter voller Jahrgang {latest_year})"
            f"</span>"
        ),
        xaxis_title="Jahr",
        yaxis_title="Temperatur-Anomalie (°C, Referenz 1961–1990)",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# -------------------------------------------------------------------
# Stations-Trends
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

    out = pd.DataFrame(
        {
            "year": year,
            "value": df[parameter],
        }
    )

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

        slope, intercept = np.polyfit(x, y, 1)
        trend_per_decade = slope * 10.0  # °C pro Dekade

        trend_rows.append(
            {
                "station_abbr": station_abbr,
                "trend_degC_per_decade": trend_per_decade,
                "n_years": df_period["year"].nunique(),
            }
        )

    trend_df = pd.DataFrame(trend_rows)
    return trend_df


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

    trend_df = trend_df.merge(
        stations[station_cols],
        on="station_abbr",
        how="left",
    )

    # Region definieren
    def classify_region(row):
        h = row["station_height_masl"]
        canton = row["station_canton"]
        if h >= 1000:
            return "Alpenraum"
        elif canton in ["TI", "VS", "GR"]:
            return "Süd"
        else:
            return "Nord"

    trend_df["region"] = trend_df.apply(classify_region, axis=1)
    return trend_df


def make_fig2(trend_df: pd.DataFrame):
    vmin = trend_df["trend_degC_per_decade"].min()
    vmax = trend_df["trend_degC_per_decade"].max()

    center_lat = trend_df["station_coordinates_wgs84_lat"].mean()
    center_lon = trend_df["station_coordinates_wgs84_lon"].mean()

    # Schönen Hover bauen
    trend_df = trend_df.copy()
    trend_df["hover"] = (
        "<b>" + trend_df["station_name"] + "</b><br>"
        + "Kanton: " + trend_df["station_canton"] + "<br>"
        + "Region: " + trend_df["region"] + "<br>"
        + "Höhe: " + trend_df["station_height_masl"].astype(int).astype(str) + " m ü. M.<br>"
        + "Trend: " + trend_df["trend_degC_per_decade"].round(2).astype(str) + " °C / Dekade<br>"
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
        title="Erwärmung pro Dekade seit 1961 an den Swiss-NBCN-Stationen",
        height=650,
    )

    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        map={"center": {"lat": center_lat, "lon": center_lon}},
        coloraxis_colorbar=dict(title="Trend<br>°C / Dekade"),
    )

    return fig

# -------------------------------------------------------------------
# Glaciers
# -------------------------------------------------------------------
def _read_shp_polygon_areas(shp_path: Path) -> float:
    """
    Reads a .shp file containing Polygon geometries and computes
    total area using the shoelace formula.

    Works only with simple SHP polygons (SGI polygons are simple).
    """
    with open(shp_path, "rb") as f:
        # skip 100-byte shapefile header
        f.read(100)

        total_area = 0.0

        while True:
            # Each record header = 8 bytes (big endian)
            header = f.read(8)
            if not header:
                break  # end of file

            # content length (in 16-bit words)
            _, content_length_words = struct.unpack(">2i", header)
            content_length_bytes = content_length_words * 2

            content = f.read(content_length_bytes)
            if not content:
                break

            # first 4 bytes = shape type (little endian)
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type != 5:  # 5 = Polygon
                continue

            # Polygon structure:
            # bytes 4–36: bounding box (ignored)
            # bytes 36–40: numParts (int)
            # bytes 40–44: numPoints (int)
            # bytes 44–...: parts indices + points

            num_parts = struct.unpack("<i", content[36:40])[0]
            num_points = struct.unpack("<i", content[40:44])[0]

            # parts array
            parts_offset = 44
            parts = []
            for i in range(num_parts):
                part_index = struct.unpack("<i", content[parts_offset + 4*i : parts_offset + 4*(i+1)])[0]
                parts.append(part_index)
            parts.append(num_points)  # final boundary end

            # points start after parts: offset = 44 + 4*num_parts
            points_offset = 44 + 4 * num_parts
            points = []
            for i in range(num_points):
                x, y = struct.unpack(
                    "<2d",
                    content[points_offset + 16*i : points_offset + 16*(i+1)]
                )
                points.append((x, y))

            # compute area of each part using shoelace formula
            for p in range(num_parts):
                start = parts[p]
                end = parts[p+1]

                pts = points[start:end]
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])

                # shoelace polygon area
                area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
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
        labels={
            "year": "Inventarjahr",
            "total_area_km2": "Gesamt-Gletscherfläche [km²]",
        },
        title="Rückgang der Gletscherfläche in der Schweiz",
    )

    # Turn line chart into an area chart
    fig.update_traces(
        line=dict(width=3),
        fill="tozeroy",  # fill from line down to zero
    )

    # Ensure y-axis always begins at zero
    fig.update_yaxes(range=[0, glaciers_df["total_area_km2"].max() * 1.05])

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# -------------------------------------------------------------------
# Snow days in Switzerland (MeteoSwiss Open Data)
# -------------------------------------------------------------------

def load_snow_days_local(base_dir: Path = SNOW_DIR) -> pd.DataFrame:
    """
    Loads all CSV files inside data/snow/normwerte/ and extracts yearly
    snow-day data for all Switzerland.

    We expect the CSVs to contain at least columns like:
        year, days, or
        YR, VALUE, or
        similar.

    The function automatically detects the correct columns.
    """

    if not base_dir.exists():
        print(f"Snow directory not found: {base_dir}")
        return pd.DataFrame(columns=["year", "snow_days"])

    rows = []

    for csv_file in sorted(base_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        df.columns = [c.lower() for c in df.columns]

        # Try common column names used by MeteoSwiss
        # (different datasets name them differently)
        year_cols = [c for c in df.columns if c.startswith("year") or c in ("yr",)]
        snow_cols = [c for c in df.columns if "snow" in c or "tage" in c or "value" in c or "days" in c]

        if not year_cols or not snow_cols:
            # Skip irrelevant files
            continue

        year_col = year_cols[0]
        snow_col = snow_cols[0]

        sub = df[[year_col, snow_col]].dropna()
        sub = sub.rename(columns={year_col: "year", snow_col: "snow_days"})

        # Keep only numeric + valid years
        sub = sub[sub["year"].astype(str).str.len() == 4]
        sub["year"] = sub["year"].astype(int)
        sub["snow_days"] = pd.to_numeric(sub["snow_days"], errors="coerce")

        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["year", "snow_days"])

    final = pd.concat(rows, ignore_index=True)

    # Remove duplicates by taking mean per year (common for station aggregates)
    final = (
        final.groupby("year")["snow_days"]
        .mean()
        .reset_index()
        .sort_values("year")
    )

    return final


def make_fig_snow_days(snow_df: pd.DataFrame):
    """
    Creates an area chart of snow days per year for Switzerland.
    """
    if snow_df.empty:
        return None

    fig = px.line(
        snow_df,
        x="year",
        y="snow_days",
        markers=True,
        labels={
            "year": "Jahr",
            "snow_days": "Schneetage (Schweiz gesamt)",
        },
        title="Anzahl Schneetage pro Jahr in der Schweiz",
    )

    # Area chart styling
    fig.update_traces(
        line=dict(width=3),
        fill="tozeroy",
    )

    fig.update_yaxes(range=[0, snow_df["snow_days"].max() * 1.05])
    fig.update_layout(template="plotly_white", hovermode="x unified")

    return fig

# -------------------------------------------------------------------
# Snow days from NIME station files (hns000d0 > 0)
# -------------------------------------------------------------------
def load_nime_station_meta() -> pd.DataFrame:
    """
    Loads NIME station metadata and returns a DataFrame with
    columns: station (abbr), height_masl, region_class (Mittelland/Alpen).
    """
    try:
        meta = pd.read_csv(NIME_META_STATIONS_URL, sep=";", encoding="latin1")
    except Exception:
        return pd.DataFrame(columns=["station", "height_masl", "region_class"])

    cols_lower = {c.lower(): c for c in meta.columns}

    # station abbreviation
    if "nat_abbr" in cols_lower:
        station_col = cols_lower["nat_abbr"]
    elif "station_abbr" in cols_lower:
        station_col = cols_lower["station_abbr"]
    else:
        # fallback: first column
        station_col = meta.columns[0]

    # height / altitude column – be defensive
    height_candidates = [
        c
        for c in meta.columns
        if any(k in c.lower() for k in ["height", "hoehe", "altitude"])
    ]
    if height_candidates:
        height_col = height_candidates[0]
    else:
        return pd.DataFrame(columns=["station", "height_masl", "region_class"])

    out = pd.DataFrame(
        {
            "station": meta[station_col].astype(str).str.upper().str.strip(),
            "height_masl": pd.to_numeric(meta[height_col], errors="coerce"),
        }
    ).dropna(subset=["height_masl"])

    # Region: simple height-based split
    #   < 1000 m  → Mittelland (inkl. tiefere Lagen / Voralpen)
    #   ≥ 1000 m  → Alpen
    def classify(h):
        if h >= 1000:
            return "Alpen"
        else:
            return "Mittelland"

    out["region_class"] = out["height_masl"].apply(classify)
    return out

def load_snow_days_from_nime(base_dir: Path = SNOW_DIR) -> pd.DataFrame:
    """
    Reads all ogd-nime_*_d_*.csv files in data/snow/ and computes:

    - Für jede Station und jedes Jahr: Anzahl Tage mit Schneehöhe > 0 cm
      (Spalte 'hns000d0', fehlende Werte als 0).
    - Verknüpft diese Zeitreihen mit der Stationshöhe aus der NIME-Metadatei.
    - Bildet zwei Gruppen:
        * Mittelland  (Stationen < 1000 m ü. M.)
        * Alpen       (Stationen ≥ 1000 m ü. M.)
    - Wählt pro Region nur Langzeit-Stationen (mind. 80 % der Jahre zwischen
      1931 und letztem Jahr vorhanden).
    - Mittelt die Schneetage pro Jahr getrennt nach Mittelland und Alpen.

    Returns DataFrame mit Spalten:
        year,
        mean_snow_days_mittelland,
        mean_snow_days_alpen,
        n_stations_mittelland,
        n_stations_alpen
    """
    if not base_dir.exists():
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_mittelland",
                "mean_snow_days_alpen",
                "n_stations_mittelland",
                "n_stations_alpen",
            ]
        )

    files = sorted(base_dir.glob("ogd-nime_*_d_*.csv"))
    if not files:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_mittelland",
                "mean_snow_days_alpen",
                "n_stations_mittelland",
                "n_stations_alpen",
            ]
        )

    meta = load_nime_station_meta()
    if meta.empty:
        # ohne Metadaten keine Höhen-Klassifikation möglich
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_mittelland",
                "mean_snow_days_alpen",
                "n_stations_mittelland",
                "n_stations_alpen",
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

        # Stationenkürzel: aus Spalte oder Dateiname
        if "station_abbr" in df.columns:
            station = df["station_abbr"].astype(str).str.upper().iloc[0]
        else:
            parts = path.stem.split("_")
            station = parts[2].upper() if len(parts) >= 3 else "UNK"

        # Zeit → Jahr
        df["reference_timestamp"] = pd.to_datetime(
            df["reference_timestamp"], dayfirst=True, errors="coerce"
        )
        df = df.dropna(subset=["reference_timestamp"])
        df["year"] = df["reference_timestamp"].dt.year

        # Schneehöhe, fehlende als 0
        fresh = pd.to_numeric(df["hns000d0"], errors="coerce")
        df["snow_day"] = fresh >= 1.0

        # Missing depth stays NaN → ignored in yearly sums
        df = df.dropna(subset=["snow_day"])

        # Anzahl Schneetage pro Jahr und Station
        per_year = (
            df.groupby("year")["snow_day"]
            .sum()
            .reset_index()
        )
        per_year["station"] = station
        per_station_year.append(per_year)

    if not per_station_year:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_mittelland",
                "mean_snow_days_alpen",
                "n_stations_mittelland",
                "n_stations_alpen",
            ]
        )

    all_sy = pd.concat(per_station_year, ignore_index=True)

    # Auf Jahre ab 1931 beschränken
    all_sy = all_sy[all_sy["year"] >= 1931]
    if all_sy.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "mean_snow_days_mittelland",
                "mean_snow_days_alpen",
                "n_stations_mittelland",
                "n_stations_alpen",
            ]
        )

    # Metadaten anhängen (Höhe und Region)
    all_sy = all_sy.merge(meta, on="station", how="left").dropna(
        subset=["height_masl", "region_class"]
    )

    start_year = 1931
    end_year = int(all_sy["year"].max())
    full_years = end_year - start_year + 1

    summaries = []

    for region_name, region_label in [
        ("Mittelland", "mittelland"),
        ("Alpen", "alpen"),
    ]:
        reg = all_sy[all_sy["region_class"] == region_name].copy()
        if reg.empty:
            continue

        # Langzeit-Stationen in dieser Region (≥80 % der Jahre vorhanden)
        cov = (
            reg.groupby("station")["year"]
            .nunique()
            .reset_index(name="n_years")
        )
        cov["coverage"] = cov["n_years"] / full_years
        long_term = cov.loc[cov["coverage"] >= 0.8, "station"]

        reg_long = reg[reg["station"].isin(long_term)]
        if reg_long.empty:
            # Fallback: wenn keine Langzeit-Stationen, nimm alle in der Region
            reg_long = reg

        # Mittel der Schneetage pro Jahr in dieser Region
        reg_summary = (
            reg_long.groupby("year")["snow_day"]
            .agg(["mean", "count"])
            .reset_index()
        )
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
                "mean_snow_days_mittelland",
                "mean_snow_days_alpen",
                "n_stations_mittelland",
                "n_stations_alpen",
            ]
        )

    # Jahreswerte der zwei Regionen zusammenführen
    out = summaries[0]
    for extra in summaries[1:]:
        out = out.merge(extra, on="year", how="outer")

    out = out.sort_values("year").reset_index(drop=True)
    return out


def make_fig_snow_days(snow_df: pd.DataFrame):
    """
    Zwei Kurven: mittlere Schneetage pro Jahr
    – getrennt nach Mittelland und Alpen.
    """
    if snow_df.empty:
        return None

    plot_df = snow_df.copy()

    # Schönere Legendennamen
    rename_map = {}
    if "mean_snow_days_mittelland" in plot_df.columns:
        rename_map["mean_snow_days_mittelland"] = "Mittelland"
    if "mean_snow_days_alpen" in plot_df.columns:
        rename_map["mean_snow_days_alpen"] = "Alpen"

    plot_df = plot_df.rename(columns=rename_map)

    y_cols = [c for c in ["Mittelland", "Alpen"] if c in plot_df.columns]
    if not y_cols:
        return None

    fig = px.line(
        plot_df,
        x="year",
        y=y_cols,
        markers=True,
        labels={
            "year": "Jahr",
            "value": "Durchschnittliche Schneetage pro Station",
            "variable": "Region",
        },
        title="Durchschnittliche Schneetage pro Jahr – Mittelland vs. Alpen",
    )

    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
    )

    # y-Achse bei 0 starten
    max_val = 0
    for col in y_cols:
        max_val = max(max_val, plot_df[col].max())
    fig.update_yaxes(range=[0, max_val * 1.05])

    return fig



# -------------------------------------------------------------------
# Convenience: alles vorbereiten
# -------------------------------------------------------------------
def prepare_all():
    # CH-Mittel
    ch = load_swiss_mean()
    ch, warming_since_start, latest_year = compute_ch_anomalies(ch)
    fig1 = make_fig1(ch, warming_since_start, latest_year)

    # Stations-Trends
    trend_df = compute_trend_df()
    trend_df = enrich_trend_with_meta(trend_df)
    fig2 = make_fig2(trend_df)

    # Glaciers
    glaciers_df = load_glacier_inventory_totals()
    fig3 = make_fig_glaciers(glaciers_df) if not glaciers_df.empty else None

    # Snow
    snow_df = load_snow_days_from_nime()
    fig4 = make_fig_snow_days(snow_df)

    return (
        ch, fig1,
        trend_df, fig2,
        warming_since_start, latest_year,
        glaciers_df, fig3,
        snow_df, fig4,
    )
