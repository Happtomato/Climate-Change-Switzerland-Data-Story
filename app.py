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
    season_balance_df,
    fig8,
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
    st.plotly_chart(fig1, width="stretch")

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
    st.plotly_chart(fig2, width="stretch")

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
        st.plotly_chart(fig3, width="stretch")
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
    st.plotly_chart(fig4, width="stretch")
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
    st.plotly_chart(fig6, width="stretch")
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

Taken together, the panels on this page point to the same conclusion.
Switzerland’s climate is shifting upward in the mean, across the map, in snow reliability, in glacier stability, and in the hottest days of the year.
And because the country’s geography links mountains and lowlands, these changes don’t stay confined to one region: they cascade.
"""
)

st.markdown("---")

# -------------------------------------------------------------------
# Section 6 Tourism
# -------------------------------------------------------------------
st.subheader("6 · Tourism patterns: winter resorts and year-round cantons")

st.markdown(
    """
Tourism is one of the clearest examples of how climate signals translate into everyday life and economic activity.
But tourism does not respond to temperature alone. Demand is shaped by many factors (prices, infrastructure, global travel trends),
so the goal here is not to “prove” a single cause, but to look for patterns that are consistent with the climate changes shown above.

To make seasonality easy to read, the chart summarizes each canton and year with a single indicator:

- **Red** means a canton is **more summer-oriented** in that year.
- **Blue** means a canton is **more winter-oriented** in that year.
- **White** means tourism is **roughly balanced** between summer and winter.

Cantons are sorted from **more summer-heavy (top)** to **more winter-heavy (bottom)** based on their long-run average.
"""
)

st.caption("Red = more summer tourism • Blue = more winter tourism • White ≈ balanced")

if fig8 is not None:
    st.plotly_chart(fig8, width="stretch")
else:
    st.warning("Tourism figure is empty. Showing raw tourism data preview.")
    st.dataframe(season_balance_df.head(50))

st.markdown(
    """
Several patterns stand out immediately.

Warm and lake-oriented destinations such as **Ticino** tend to remain strongly **summer-dominated**.
In contrast, alpine cantons that historically rely more on snow-based activities show a more **winter-oriented** profile.
This difference is expected: the tourism “climate” of a canton is shaped by elevation, landscape, and what visitors come for.

The second layer is time. In many cantons, the balance becomes slightly more **summer-leaning** in recent years.
This should be interpreted carefully: it does not necessarily mean winter tourism is collapsing everywhere.
Rather, it suggests that **summer has gained relative weight** and winter has become **more variable** a pattern that fits with the
warming trend and the declining reliability of snow shown earlier in this story.
"""
)

st.caption(
    "Data: Swiss Federal Statistical Office (BFS), STAT-TAB PxWeb "
    "table px-x-1003020000_102 (hotel overnight stays, total visitors)."
)

st.markdown("---")

# -------------------------------------------------------------------
# Conclusion
# -------------------------------------------------------------------
st.subheader("Conclusion · One climate signal, many connected changes")

st.markdown(
    f"""
This data story started with a long temperature record and ended with a set of consequences that connect directly to Switzerland’s landscapes
and to daily life.

The Swiss mean temperature series shows a clear shift toward a warmer climate, reaching about **{warming_since_start:.1f} °C** of warming
between the earliest decades of measurements and the most recent decade. That signal is not limited to one place: the station-based trends
show warming across the country.

From there, the impacts become easier to see. Snow is highly sensitive around the freezing point, so even modest warming changes whether precipitation
falls as snow or rain and how long snow persists. In the snow-day record, this appears as declining snow reliability, especially at lower elevations.
Glaciers respond more slowly, but they integrate many seasons at once and the glacier inventory shows a continued decline in total glacier area,
which is difficult to reverse on human timescales.

Extremes matter too. Rising maximum temperatures raise the likelihood of heat stress in populated lowlands and push warmth into higher terrain,
affecting ecosystems and accelerating melt processes. Finally, the tourism seasonality analysis adds a human dimension:
some cantons remain structurally summer-oriented, while alpine regions show stronger winter dependence, and recent years in many cantons lean slightly
more toward summer activity.

No single diagram tells the full story. But together, temperature trends, snow, glaciers, extremes, and tourism point in the same direction:
Switzerland’s climate baseline has shifted, and the effects cascade through both natural systems and climate-sensitive sectors.
The key challenge going forward is not recognizing the signal it is adapting infrastructure, risk management, and economic planning to a climate
that is no longer the one Switzerland was built around.
"""
)
