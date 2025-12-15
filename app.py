# =================
# app.py (ENGLISH) more narrative
# =================
import streamlit as st
from climate_core import prepare_all

# Prepare data + figures (computed once per run)
(
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
) = prepare_all()

# -------------------------------------------------------------------
# Streamlit layout
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Switzerland is getting warmer and the Alps are changing",
    layout="wide",
)

# -------------------------------------------------------------------
# Hero section
# -------------------------------------------------------------------
st.title("Switzerland is getting warmer and the Alps are changing")
st.markdown("### Temperature, snow, and glaciers in transition since 1864.")

st.markdown(
    """
Switzerland is often imagined as a country defined by stable seasons: crisp winters, reliable snow in the mountains, and mild summers in the lowlands.
But the climate that shaped those expectations is shifting. The change is not abstract it shows up in the long temperature record, in the
geography of warming across the country, and in the visible retreat of snow and ice.

This page tells one connected story: as temperatures rise, the Alps respond first and strongly, snow becomes less reliable at lower elevations,
and glaciers which accumulate decades of climate in a single signal keep shrinking.
"""
)

st.markdown(
    """
**Key message:** Switzerland is warming fast, and the consequences cascade through the mountain environment affecting water, ecosystems, tourism,
and natural hazards.
"""
)

st.markdown("---")

# -------------------------------------------------------------------
# Section 1 Time series
# -------------------------------------------------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("1 · A long record with a clear direction")
    st.markdown(
        f"""
The Swiss temperature record is one of the most valuable ways to “see” climate change, because it spans many generations.
The curve below shows annual temperature anomalies relative to the 1961–1990 reference period. In other words, it answers a simple question:
*How different is each year compared to a climate that used to feel normal?*

When you look at the full time span, the most striking feature is not any single hot year it’s the steady lift of the entire distribution.
Warm years become more frequent, and cold years become rarer. Over the period covered here, the country ends up about **{warming_since_start:.1f} °C**
warmer than the earliest decades of measurements.

Certain summers are still worth highlighting, because they anchor the story in lived experience. Years like **2003, 2015, and 2022** stand out as
spikes but they also hint at something deeper: what once counted as “extreme” is showing up more often, and it happens on top of a higher baseline.
"""
    )

    st.markdown(
        f"""
The thicker line is an 11-year running mean. It smooths out short-term variability and makes the long-term signal easier to read.
By the most recent full year (**{latest_year}**), the trend is unmistakable: Switzerland’s climate has moved into a warmer regime.
"""
    )

with col2:
    st.plotly_chart(fig1, use_container_width=True)

st.caption(
    "Data: homogenized Swiss mean temperature. Baseline: 1961–1990. "
    "Anomalies are deviations from the mean of that baseline."
)

st.markdown("---")

# -------------------------------------------------------------------
# Section 2 Map
# -------------------------------------------------------------------
col1, col2 = st.columns([1, 1.1])

with col1:
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("2 · The geography of warming")
    st.markdown(
        """
A national average is useful, but it can hide the way climate change plays out across different landscapes.
Switzerland is especially interesting because it compresses strong elevation differences into a small area.
That makes it a natural laboratory for a key question: *Does warming look the same in the lowlands and in the mountains?*

Each point on the map is a Swiss NBCN station. Its color indicates the warming trend per decade since 1961.
Even without over-interpreting any single station, the overall picture is coherent: the country warms across the board.
The differences that remain are not about whether warming exists, but about where it is strongest and how it is experienced.

In higher terrain, warming often comes with feedbacks. When snow and ice retreat, darker ground is exposed, and the surface can absorb more sunlight.
That can amplify local warming and shift the balance of the seasons. In the lowlands, the warming signal tends to show up as hotter summers and
milder winters changes that affect health, agriculture, and cities through heat stress.
"""
    )

    st.markdown(
        """
Rather than reading the map as a competition between regions, it helps to see it as one connected system.
The Alps, the plateau, and the southern valleys are linked by weather patterns, water flow, and infrastructure.
A warmer climate in one part of the country inevitably influences the others.
"""
    )

st.caption(
    "Data: homogenized annual mean temperature (ths200y0). Linear trends over 1961–2024 in °C per decade."
)

st.markdown("---")

# -------------------------------------------------------------------
# Section 3 Glaciers
# -------------------------------------------------------------------
st.subheader("3 · Glaciers as the landscape’s memory")

col1, col2 = st.columns([1.1, 1])

with col1:
    if fig3 is not None:
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No glacier shapefiles found.")

with col2:
    st.markdown(
        """
Glaciers respond to climate in a way that feels almost personal: they shrink, fragment, and retreat until entire ice bodies disappear.
Unlike a single hot day, glacier change integrates many seasons at once. That makes glaciers a kind of long-term memory stored in the landscape.

The chart shows the total glacier area in Switzerland at different inventory years. Each point is a snapshot, taken with careful mapping,
and together they form a simple narrative: the baseline is falling.

What does that mean in practice? For a while, stronger melt can increase runoff during warm seasons, because more ice is turning into water.
But as glacier volume declines, that “extra” water becomes a temporary benefit that fades. Over the long run, the loss of ice reduces the
buffer that helps stabilize summer water supply during dry periods.

The retreat also reshapes risk. As ice disappears, new meltwater lakes can form behind unstable natural dams. Slopes that were once supported by
permafrost can destabilize as ground temperatures rise. The result is not just a change in scenery it’s a shift in how mountain landscapes behave.
"""
    )

    st.markdown(
        """
A useful way to think about glaciers is that they turn warming into something visible and cumulative. They do not rebound quickly.
So when glacier area drops, it is a strong signal that the climate has moved beyond the conditions that sustained that ice.
"""
    )

st.caption(
    "Data: Swiss Glacier Inventory (SGI). Total area summed across all glaciers per inventory year."
)

st.markdown("---")

# -------------------------------------------------------------------
# Section 4 Snow days
# -------------------------------------------------------------------
st.subheader("4 · Snow: the season that is slipping")

if fig4 is not None:
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("No snow-day data found in 'data/snow/'.")

st.markdown(
    """
Snow is one of the most culturally and economically important signals of Alpine climate. It shapes winter tourism, influences hydrology,
and defines what many people expect from the mountains. But snow is also extremely sensitive to temperature: a small shift around the freezing point
can flip precipitation from snow to rain, and it can shorten the season even when total precipitation stays similar.

This analysis counts the number of days per year with fresh snowfall (new snow height above zero, measured from 06 UTC to 06 UTC the next day).
Missing observations are ignored rather than treated as zero, so the trend is not artificially pushed downward by gaps in the record.
To keep the storyline readable, the station results are aggregated into two elevation bands: **Lowlands** (below 1000 m) and **Alps** (1000 m and above).

In the lowlands, the decline in snow days is often felt first. Winters become more marginal, and snow shifts from a dependable season to an intermittent event.
In the Alps, snow remains more common, but the direction still matters: fewer snow days can mean earlier melt, thinner snowpack, and more frequent winter rain,
all of which affects ecosystems, transport, and hazard management.
"""
)

st.markdown(
    """
It is tempting to focus only on iconic high-elevation resorts, but the lowlands are where most people live.
When snow days fall there, it changes everyday experience and it also changes how water moves through the year, because more winter precipitation arrives as rain
and flows out quickly instead of being stored as snowpack.
"""
)

st.markdown("---")

# -------------------------------------------------------------------
# Section 5 Extreme heat
# -------------------------------------------------------------------
st.subheader("5 · When the hottest day gets hotter")

if fig6 is not None:
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.warning("No extreme temperature data available.")

st.markdown(
    """
Average warming is the background story. Extremes are often where impacts become immediate.

The figure above tracks the annual absolute maximum temperature (measured at 2 meters above ground), averaged across stations and split by elevation band.
It captures a different aspect of climate change: not just how the typical year shifts, but how high the ceiling rises.

In the lowlands, higher extremes increase the likelihood of heat stress, especially in cities where buildings and asphalt store warmth.
In the Alps, rising extremes matter because they push heat into places and seasons that historically stayed cooler.
That can accelerate glacier melt, stress cold-adapted ecosystems, and affect infrastructure built for different temperature ranges.

Taken together, the five panels on this page point to the same conclusion.
Switzerland’s climate is shifting upward in the mean, across the map, in snow reliability, in glacier stability, and in the hottest days of the year.
And because the country’s geography links mountains and lowlands, these changes don’t stay confined to one region: they cascade.
"""
)

st.markdown(
    """
If you want a single mental image to carry forward, make it this:
a warmer baseline lifts everything the average year, the extreme years, the rain–snow boundary, and the long-term balance of ice.
Once you see those pieces together, the story becomes hard to unsee.
"""
)