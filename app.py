import streamlit as st

from climate_core import prepare_all

# Daten + Figuren vorbereiten (wird nur einmal pro Run gemacht)
ch, fig1, trend_df, fig2, warming_since_start, latest_year = prepare_all()

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

# -----------------------------------------------------------
# Abschnitt 1 – Zeitreihe
# -----------------------------------------------------------
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
# Abschnitt 3 – Platz für Schnee & Gletscher Charts deines Kollegen
# -----------------------------------------------------------
st.subheader("3 · Was bedeutet das für Schnee und Gletscher?")

st.markdown(
    """
Hier können die nächsten beiden Grafiken folgen, z.B.:

- Entwicklung von **Schneetagen / Schneehöhen** an ausgewählten Stationen  
- Rückzug der **Gletscherfläche oder Eismasse** in den letzten Jahrzehnten  

Weniger kalte Tage und kürzere Schneesaisons führen dazu, dass:

- Wintertourismus auf tieferen Lagen unter Druck gerät  
- Gletscher massiv an Volumen verlieren und im Sommer immer mehr Schmelzwasser liefern  
- sich Naturgefahren verändern – z.B. durch auftauenden Permafrost oder neue Seen an Gletscherfronten.
"""
)
