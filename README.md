# 🌞 Solvion — NASA Space Apps Challenge 2025

**Project Name:** Solar Energy Solution  
**Team:** Solvion  
**Challenge:** #2 – Solar Energy  
**Location:** Switzerland 🇨🇭  
**Event:** NASA Space Apps Challenge 2025  

---

## 🚀 Goal
Develop a data-driven solution using **NASA datasets** and **APIs** to analyze and optimize the use of **solar energy** for sustainable development and clean power generation.

---

## 🧠 Idea
Our project focuses on:
- Using **NASA POWER API** and satellite data to estimate solar potential.
- Visualizing solar irradiation and weather conditions.
- Providing insights for energy planning and renewable solutions.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, Xarray, Matplotlib, Streamlit, PyTorch

    
## 📊 Data Sources

- 🌍 **NASA POWER Data Access** – Solar & meteorological data  
  🔗 https://power.larc.nasa.gov/  

- ☁️ **MeteoSwiss Open Data** – Hourly/daily weather observations & forecasts  
  🔗 https://opendata.swiss/de  

- 🌡 **MeteoSwiss API** – Real-time forecast weather data  
  🔗 https://opendata.swiss/de  

- 🌐 **Copernicus Climate Data Store** – Reanalysis, seasonal forecasts, ERA5 datasets  
  🔗 https://cds.climate.copernicus.eu/  

- ⚡ **Swissgrid Transparency Platform** – Load, production, and balancing data for Switzerland  
  🔗 https://www.swissgrid.ch/en/home/operation/grid-data/transparency.html  

- 🔧 **pvlib-python** – Solar power modeling based on weather forecasts  
  🔗 https://pvlib-python.readthedocs.io/  

---

## Project Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📂 Project Structure


## Use Interactive Dashboard
```
pip install streamlit streamlit-folium folium meteomatics pandas numpy scikit-learn python-dotenv joblib
streamlit run dashboard/app.py
```

## 🧾 License
This project is licensed under the **MIT License**.

