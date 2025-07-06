# CrimeSight-Visual-Analytics-and-Forecasting-of-Crime-in-India
This project is a Streamlit-based web application that provides an interactive dashboard to analyze, visualize, and predict crime rates across Indian cities using real-world data.

## Features
- **Preprocessing**
  
- **Visual Analytics**
  - Crime distribution by city and type
  - Victim age distribution
  - Weapons used in crimes
  - Crime trends over time
  - Correlation heatmap of numerical features

- **Machine Learning**
  - Linear Regression model to predict future crime trends
  - Performance metrics: MAE, RMSE, MAPE

- **Interactive Filters**
  - Filter dataset by city and crime type via sidebar
  - Dynamic graphs and summaries based on filters

- **Future Predictions**
  - Predicts crime counts for the next 5 years
 
---

## Prerequisites
- Python 3.7+
- Required libraries:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - scikit-learn

---

## Install Dependencies
Use pip to install the required packages:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
```

## Run the app
```bash
streamlit run app.py
```
The app will open in your default browser at `http://localhost:8501/`

--- 

## Sample Dataset columns

- City
- Victim Age
- Victim Gender
- Weapon Used
- Crime Domain
- Date Reported
- Date of Occurrence
- Date Case Closed

---

## ML Approach

- Model: Linear Regression
- Target: Total crimes per year
- Features Used: Year
- Train/Test Split: 80/20
