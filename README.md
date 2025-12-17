# Climate Change in Switzerland – A Data Story 🇨🇭

This project is an interactive **data visualization and storytelling website** about climate change in Switzerland.
It was created as the **final project for the DVIZ (Data Visualization) course** and explores how long-term climate change affects temperature, snow, glaciers, extreme heat, and tourism patterns across the country.

The goal of the project is not only to present data, but to **tell a coherent story** that connects physical climate signals with real-world impacts and human systems.

---

## 🎓 Course Context

* **Course:** I.BA_DVIZ_MM.H2501 / DATA VISUALIZATION FOR AI AND ML
* **Project type:** Final group project
* **Institution:** Lucerne University of Applied Sciences and Arts (HSLU)

### 👥 Group Members

* **Dominik Dierberger**
* **Diego Kurz**

---

## 🌍 Project Overview

The website guides the user through several connected chapters:

1. **Long-term temperature change**
   Switzerland’s homogenized temperature record since the 19th century shows a clear and persistent warming trend.

2. **Geography of warming**
   Station-based trends reveal how warming varies across regions and elevations.

3. **Glacier retreat**
   Glacier inventories illustrate how glaciers act as long-term indicators of climate change.

4. **Snow reliability**
   Declining snow days highlight how sensitive winter conditions are to rising temperatures.

5. **Extreme heat**
   Changes in annual maximum temperatures show how extremes intensify alongside average warming.

6. **Tourism seasonality**
   Seasonal tourism patterns by canton reveal how climate-sensitive regions differ and how summer tourism has gained relative importance in recent decades.

Together, these elements form a **data-driven climate story** that links natural systems and human activity.

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit** (web app & layout)
* **Pandas / NumPy** (data processing)
* **Plotly** (interactive visualizations)

All data is processed locally from official Swiss open-data sources (MeteoSwiss, BFS, Swiss Glacier Inventory).

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Happtomato/Climate-Change-Switzerland-Data-Story
cd Climate-Change-Switzerland-Data-Story
```

### 2️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Start the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically in your browser (usually at `http://localhost:8501`).

---

## 📁 Project Structure

```
Climate-Change-Switzerland-Data-Story/
├── app.py                 # Streamlit app (main entry point)
├── climate_core.py        # Data loading, processing, and figure creation
├── download.py            # Download MeteoSwiss NBCN metadata
├── download2.py           # Download MeteoSwiss yearly station data
├── download3.py           # Download snow data (NIME stations)
├── download4.py           # Download SMN temperature extremes
├── download5.py           # Download BFS tourism data (hotel overnight stays)
├── data/                  # All downloaded datasets (used by the app)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🐍 Environment & Requirements

This project was developed and tested with:

* **Python:** 3.13
* **Streamlit:** ≥ 1.32
* **Pandas:** ≥ 2.0
* **NumPy:** ≥ 1.26
* **Plotly:** ≥ 5.18
* **Requests:** ≥ 2.31

All required Python packages are listed in `requirements.txt`.

> Note: The project should also run with other recent Python 3 versions, but Python 3.13 was used during development.

---

## 📊 Data Sources

* **MeteoSwiss** – homogenized temperature records, station data, snow observations
* **Swiss Glacier Inventory (SGI)** – glacier area inventories
* **Swiss Federal Statistical Office (BFS)** – hotel overnight stays by canton

All data sources are publicly available and used for educational purposes.

---

## 🎯 Purpose & Disclaimer

This project is **educational** and created for a university course.
It aims to visualize and contextualize climate trends, not to provide predictions or policy recommendations.

Observed patterns  especially in tourism  should be interpreted carefully, as climate change is only one of several influencing factors.