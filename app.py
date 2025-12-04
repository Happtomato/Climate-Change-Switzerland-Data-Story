import streamlit as st

from climate_core import prepare_all

# Daten + Figuren vorbereiten (wird nur einmal pro Run gemacht)
(
    ch, fig1,
    trend_df, fig2,
    warming_since_start, latest_year,
    glaciers_df, fig3,
    snow_df, fig4,
) = prepare_all()

# -----------------------------------------------------------
# Streamlit Layout
# -----------------------------------------------------------
st.set_page_config(
    page_title="Die Schweiz wird wärmer – und die Alpen verändern sich",
    layout="wide",
)

# Hero-Section
st.title("Die Schweiz wird wärmer – und die Alpen verändern sich")
st.markdown("### Temperatur, Schnee und Gletscher im Wandel seit 1864.")

st.markdown(
    """
**Kernbotschaft**

- Die Schweiz erwärmt sich stärker als der globale Durchschnitt.  
- Die Erwärmung verläuft regional und mit der Höhe unterschiedlich stark.  
- Weniger Kälte bedeutet weniger Schneetage und starke Gletscherschmelze.  
- Das verändert Natur und Gesellschaft – von Wasserressourcen über Tourismus bis hin zu Naturgefahren.
"""
)

st.markdown("---")

# ----------------------------------------------------------
# Abschnitt 1 – Zeitreihe
# ----------------------------------------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("1 · Wie stark hat sich die Schweiz erwärmt?")
    st.markdown(
        f"""
Seit Beginn der flächendeckenden Messungen (**1864**) ist die mittlere Jahres-Temperatur der Schweiz
deutlich gestiegen.

- Im Vergleich zu den ersten Messjahrzehnten liegt die Schweiz heute um rund **{warming_since_start:.1f} °C** höher.  
- Hitzesommer wie **2003, 2015 oder 2022** stechen als Extremjahre hervor – sie liegen weit über der Referenzperiode 1961–1990.  
- Die geglättete Kurve zeigt: Es handelt sich nicht um zufällige Ausschläge, sondern um einen **klaren, langfristigen Erwärmungstrend**.

Die Schweiz erwärmt sich damit **stärker als der globale Durchschnitt** – ein typisches Muster in Gebirgsregionen
mit viel Schnee und Gletschern.
"""
    )

with col2:
    st.plotly_chart(fig1, width="stretch")

st.caption(
    "Daten: Homogenisierte Schweizer Mitteltemperatur, Referenzperiode 1961–1990. "
    "Anomalien = Abweichung vom Mittel dieser Referenzperiode."
)

st.markdown("---")

# -----------------------------------------------------------
# Abschnitt 2 – Karte
# -----------------------------------------------------------
col1, col2 = st.columns([1, 1.1])

with col1:
    st.plotly_chart(fig2, width="stretch")

with col2:
    st.subheader("2 · Die Erwärmung ist nicht überall gleich")
    st.markdown(
        """
Die Karte zeigt die **Temperaturtrends seit 1961** an ausgewählten Messstationen des Swiss NBCN-Netzes.
Die Farbe gibt an, wie stark sich die Jahresmitteltemperatur pro Dekade verändert hat.

Wichtige Muster:

- **Alpenraum:** Viele Stationen in höherer Lage zeigen die **stärksten Erwärmungsraten**.
  In Gebirgen verstärkt sich der Klimawandel oft – unter anderem, weil Schnee- und Eisbedeckung
  zurückgehen und der Boden dadurch mehr Sonnenenergie aufnimmt.
- **Mittelland:** Auch hier ist der Trend klar positiv – die Sommer werden heißer, die Winter milder.
- **Südseite der Alpen:** Erwärmung ebenfalls deutlich, oft kombiniert mit längeren Trockenphasen.

Insgesamt gibt es keine Station mit „Nulltrend“ – **überall** in der Schweiz zeigen die Messreihen
einen **klaren Anstieg** der Temperatur.
"""
    )

st.caption(
    "Daten: Homogenisierte Jahresmitteltemperatur ths200y0, lineare Trends 1961–2024 in °C pro Dekade."
)

st.markdown("---")

# -----------------------------------------------------------
# Abschnitt 3 – Schnee & Gletscher
# -----------------------------------------------------------
st.subheader("3 · Was bedeutet das für Schnee und Gletscher?")

col1, col2 = st.columns([1.1, 1])

with col1:
    if fig3 is not None:
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Keine Gletscher-Shapefiles gefunden.")

with col2:
    st.markdown(
        """
Die Grafik zeigt die **Gesamtfläche aller Gletscher in der Schweiz** pro Inventarjahr.
Jeder Punkt entspricht einer vollständigen Aufnahme des Schweizer Gletscherinventars
(z. B. **1931**).

Die Botschaft:

- Selbst über wenige Inventare hinweg ist ein **klarer Rückgang der Gletscherfläche** sichtbar.  
- Wärmere Jahre bedeuten eine **kürzere Schneesaison** und mehr Schmelze im Sommer.  
- Langfristig verschwinden kleinere Gletscher ganz, größere ziehen sich stark zurück.

Das hat Folgen für:

- **Wasserhaushalt:** mehr Schmelzwasser im Sommer heute, weniger Eisreserve in Zukunft  
- **Tourismus:** Skigebiete auf tieferen Lagen verlieren an Schneesicherheit  
- **Naturgefahren:** neue Gletscherseen, instabile Hänge und auftauender Permafrost
        """
    )

st.caption(
    "Daten: Schweizer Gletscherinventar (SGI), Summen der Flächen aller Gletscher pro Inventarjahr."
)
# -----------------------------------------------------------
# Abschnitt 4 – Schneetage in der Schweiz
# -----------------------------------------------------------
st.subheader("4 · Wie verändern sich die Schneetage in der Schweiz?")

if fig4 is not None:
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("Keine Schneetage-Daten in 'data/snow/' gefunden.")

st.markdown(
    """
Hier siehst du die **durchschnittliche Anzahl von Tagen pro Jahr**, an denen
die gemessene Schneehöhe um 6 UTC grösser als 0 cm war.

Vorgehen:

- Für jede Station und jedes Jahr wird gezählt, an wie vielen Tagen `gemessene Schneehöhe um 6 UTC > 0` war  
  (fehlende Werte werden als 0 behandelt).  
- Diese Anzahl wird dann über alle verfügbaren Stationen gemittelt.

So erkennst du, wie stark die **Schneetage im Mittel abnehmen** – ergänzend zum Rückgang
der Gletscherfläche aus der Grafik darüber.
"""
)
