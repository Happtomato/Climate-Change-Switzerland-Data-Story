from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -------------------------------------------------------------------
# Pfade & Konstanten
# -------------------------------------------------------------------
DATA_DIR = Path("data")
SWISS_MEAN_PATH = DATA_DIR / "climate-data-swissmean_regSwiss_1.4.txt"

METEOSWISS_META_DIR = DATA_DIR / "meteoswiss_nbcn_meta"
METEOSWISS_YEARLY_DIR = DATA_DIR / "meteoswiss_nbcn_yearly"

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

    return ch, fig1, trend_df, fig2, warming_since_start, latest_year
