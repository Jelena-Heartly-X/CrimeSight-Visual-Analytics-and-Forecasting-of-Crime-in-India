import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, confusion_matrix, roc_auc_score)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Crime Analytics System", layout="wide", initial_sidebar_state="expanded")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #2ca02c; margin-top: 20px;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .insight-box {background-color: #e8f4f8; padding: 15px; border-left: 5px solid #1f77b4; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING & PREPROCESSING ====================
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the crime dataset with advanced feature engineering"""
    df = pd.read_csv('crime_dataset_india.csv')
    
    # Convert dates
    df['Date Reported'] = pd.to_datetime(df['Date Reported'], errors='coerce', dayfirst=True)
    df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], errors='coerce', dayfirst=True)
    df['Date Case Closed'] = pd.to_datetime(df['Date Case Closed'], errors='coerce', dayfirst=True)
    df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], errors='coerce', format='%d-%m-%Y %H:%M')
    
    # Advanced Feature Engineering
    df['Year'] = df['Date Reported'].dt.year
    df['Month'] = df['Date Reported'].dt.month
    df['Day'] = df['Date Reported'].dt.day
    df['DayOfWeek'] = df['Date Reported'].dt.dayofweek
    df['Quarter'] = df['Date Reported'].dt.quarter
    df['Hour'] = df['Time of Occurrence'].dt.hour
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Season mapping
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['Season'] = df['Month'].map(season_map)
    
    # Case resolution metrics
    df['Time to Close Case'] = (df['Date Case Closed'] - df['Date Reported']).dt.days
    df['Case Closed Binary'] = df['Case Closed'].map({'Yes': 1, 'No': 0})
    df['Resolution Efficiency'] = np.where(df['Time to Close Case'] <= 30, 'Fast',
                                           np.where(df['Time to Close Case'] <= 90, 'Medium', 'Slow'))
    
    # Crime severity classification
    violent_crimes = ['HOMICIDE', 'ASSAULT', 'KIDNAPPING', 'ROBBERY']
    df['Crime Severity'] = df['Crime Description'].apply(
        lambda x: 'High' if any(crime in str(x).upper() for crime in violent_crimes) else 'Medium'
    )
    
    return df

# ==================== RISK SCORING SYSTEM ====================
def calculate_risk_scores(df):
    """Calculate risk scores for cities and crime types"""
    
    # City Risk Score (weighted combination)
    city_metrics = df.groupby('City').agg({
        'Report Number': 'count',
        'Crime Severity': lambda x: (x == 'High').sum(),
        'Case Closed Binary': 'mean',
        'Time to Close Case': 'mean'
    }).reset_index()
    
    city_metrics.columns = ['City', 'Total Crimes', 'High Severity Crimes', 'Resolution Rate', 'Avg Resolution Time']
    
    # Normalize and calculate risk score
    scaler = StandardScaler()
    city_metrics['Crime Volume Score'] = scaler.fit_transform(city_metrics[['Total Crimes']])
    city_metrics['Severity Score'] = scaler.fit_transform(city_metrics[['High Severity Crimes']])
    city_metrics['Resolution Score'] = -scaler.fit_transform(city_metrics[['Resolution Rate']])
    
    city_metrics['Risk Score'] = (
        0.4 * city_metrics['Crime Volume Score'] +
        0.3 * city_metrics['Severity Score'] +
        0.3 * city_metrics['Resolution Score']
    )
    
    # Normalize to 0-100 scale
    city_metrics['Risk Score'] = ((city_metrics['Risk Score'] - city_metrics['Risk Score'].min()) / 
                                   (city_metrics['Risk Score'].max() - city_metrics['Risk Score'].min()) * 100)
    
    return city_metrics.sort_values('Risk Score', ascending=False)

# ==================== MAIN APP ====================
st.markdown('<p class="main-header">🚨 Advanced Crime Analytics & Risk Management System</p>', unsafe_allow_html=True)
st.markdown("**Enterprise-Grade Predictive Analytics for Law Enforcement Resource Optimization**")

# Load data
try:
    df = load_and_preprocess_data()
    st.success(f"✅ Dataset loaded successfully: {len(df):,} records")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ==================== SIDEBAR FILTERS ====================
st.sidebar.header("🔍 Advanced Filters")
selected_cities = st.sidebar.multiselect("Select Cities", options=sorted(df["City"].dropna().unique()), 
                                         default=sorted(df["City"].dropna().unique())[:3])
selected_crimes = st.sidebar.multiselect("Select Crime Types", 
                                         options=sorted(df["Crime Description"].dropna().unique()),
                                         default=sorted(df["Crime Description"].dropna().unique())[:5])
date_range = st.sidebar.date_input("Date Range", 
                                   value=(df['Date Reported'].min(), df['Date Reported'].max()),
                                   min_value=df['Date Reported'].min(),
                                   max_value=df['Date Reported'].max())

# Apply filters
filtered_df = df.copy()
if selected_cities:
    filtered_df = filtered_df[filtered_df["City"].isin(selected_cities)]
if selected_crimes:
    filtered_df = filtered_df[filtered_df["Crime Description"].isin(selected_crimes)]
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['Date Reported'] >= pd.Timestamp(date_range[0])) & 
                              (filtered_df['Date Reported'] <= pd.Timestamp(date_range[1]))]

# ==================== EXECUTIVE DASHBOARD ====================
st.markdown('<p class="sub-header">📊 Executive Dashboard - Key Performance Indicators</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Crimes", f"{len(filtered_df):,}", 
              delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None)
with col2:
    resolution_rate = filtered_df['Case Closed Binary'].mean() * 100
    st.metric("Resolution Rate", f"{resolution_rate:.1f}%",
              delta=f"{resolution_rate - df['Case Closed Binary'].mean()*100:.1f}%")
with col3:
    avg_resolution = filtered_df['Time to Close Case'].mean()
    st.metric("Avg Resolution (days)", f"{avg_resolution:.0f}",
              delta=f"{avg_resolution - df['Time to Close Case'].mean():.0f}")
with col4:
    high_severity = (filtered_df['Crime Severity'] == 'High').sum()
    st.metric("High Severity Crimes", f"{high_severity:,}",
              delta=f"{high_severity - (df['Crime Severity'] == 'High').sum()}")
with col5:
    avg_police = filtered_df['Police Deployed'].mean()
    st.metric("Avg Police Deployed", f"{avg_police:.1f}",
              delta=f"{avg_police - df['Police Deployed'].mean():.1f}")

# ==================== RISK ANALYSIS ====================
st.markdown('<p class="sub-header">⚠️ Risk Analysis & Scoring</p>', unsafe_allow_html=True)

risk_scores = calculate_risk_scores(filtered_df)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**City Risk Scores (0-100 scale)**")
    fig = px.bar(risk_scores.head(10), x='City', y='Risk Score', 
                 color='Risk Score', color_continuous_scale='Reds',
                 title='Top 10 High-Risk Cities')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Risk Metrics Breakdown**")
    st.dataframe(risk_scores[['City', 'Total Crimes', 'High Severity Crimes', 
                              'Resolution Rate', 'Risk Score']].head(10).style.background_gradient(
                                  subset=['Risk Score'], cmap='Reds'))

# Insights
highest_risk = risk_scores.iloc[0]
st.markdown(f"""
<div class="insight-box">
<b>🎯 Key Insight:</b> {highest_risk['City']} has the highest risk score ({highest_risk['Risk Score']:.1f}) 
with {highest_risk['Total Crimes']:.0f} total crimes and a resolution rate of {highest_risk['Resolution Rate']*100:.1f}%.
Recommend increasing police deployment by 15-20% in this region.
</div>
""", unsafe_allow_html=True)

# ==================== TEMPORAL ANALYSIS ====================
st.markdown('<p class="sub-header">📈 Temporal Pattern Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Hourly patterns
    hourly_crimes = filtered_df.groupby('Hour').size().reset_index(name='Crime Count')
    fig = px.line(hourly_crimes, x='Hour', y='Crime Count', 
                  title='Crime Distribution by Hour of Day',
                  markers=True)
    fig.add_hline(y=hourly_crimes['Crime Count'].mean(), line_dash="dash", 
                  line_color="red", annotation_text="Average")
    st.plotly_chart(fig, use_container_width=True)
    
    peak_hour = hourly_crimes.loc[hourly_crimes['Crime Count'].idxmax(), 'Hour']
    st.info(f"🕐 Peak crime hour: {peak_hour}:00 - Recommend increased patrol during this time")

with col2:
    # Day of week patterns
    dow_crimes = filtered_df.groupby('DayOfWeek').size().reset_index(name='Crime Count')
    dow_crimes['Day'] = dow_crimes['DayOfWeek'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'})
    fig = px.bar(dow_crimes, x='Day', y='Crime Count',
                 title='Crime Distribution by Day of Week',
                 color='Crime Count', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# Seasonal analysis
col1, col2 = st.columns(2)

with col1:
    season_crimes = filtered_df.groupby('Season').size().reset_index(name='Crime Count')
    fig = px.pie(season_crimes, values='Crime Count', names='Season',
                 title='Seasonal Crime Distribution',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    monthly_trend = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Crime Count')
    monthly_trend['Date'] = pd.to_datetime(monthly_trend[['Year', 'Month']].assign(day=1))
    fig = px.line(monthly_trend, x='Date', y='Crime Count',
                  title='Monthly Crime Trends',
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ==================== CRIME TYPE ANALYSIS ====================
st.markdown('<p class="sub-header">🔍 Crime Type & Weapon Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    crime_severity = filtered_df.groupby(['Crime Description', 'Crime Severity']).size().reset_index(name='Count')
    top_crimes = crime_severity.nlargest(15, 'Count')
    fig = px.bar(top_crimes, x='Count', y='Crime Description', color='Crime Severity',
                 title='Top 15 Crime Types by Severity',
                 orientation='h', color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    weapon_analysis = filtered_df.groupby('Weapon Used').agg({
        'Report Number': 'count',
        'Case Closed Binary': 'mean'
    }).reset_index()
    weapon_analysis.columns = ['Weapon', 'Count', 'Resolution Rate']
    weapon_analysis = weapon_analysis.sort_values('Count', ascending=False).head(10)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Crime Count', x=weapon_analysis['Weapon'], 
                         y=weapon_analysis['Count'], yaxis='y', offsetgroup=1))
    fig.add_trace(go.Scatter(name='Resolution Rate', x=weapon_analysis['Weapon'],
                            y=weapon_analysis['Resolution Rate']*100, yaxis='y2',
                            mode='lines+markers', marker=dict(color='red', size=8)))
    
    fig.update_layout(
        title='Weapon Usage vs Resolution Rate',
        yaxis=dict(title='Crime Count'),
        yaxis2=dict(title='Resolution Rate (%)', overlaying='y', side='right'),
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== CLUSTERING ANALYSIS ====================
st.markdown('<p class="sub-header">🎯 Crime Pattern Clustering (K-Means)</p>', unsafe_allow_html=True)

cluster_features = filtered_df[['Victim Age', 'Police Deployed', 'Hour']].dropna()
if len(cluster_features) > 10:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # Optimal clusters using elbow method
    inertias = []
    K_range = range(2, min(8, len(cluster_features)//10))
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers'))
        fig.update_layout(title='Elbow Method for Optimal K',
                         xaxis_title='Number of Clusters',
                         yaxis_title='Inertia',
                         height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_features['Cluster'] = kmeans.fit_predict(scaled_features)
        
        fig = px.scatter_3d(cluster_features, x='Victim Age', y='Police Deployed', z='Hour',
                           color='Cluster', title=f'Crime Clusters (K={optimal_k})',
                           labels={'Cluster': 'Pattern Type'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster interpretation
    cluster_summary = cluster_features.groupby('Cluster').agg({
        'Victim Age': 'mean',
        'Police Deployed': 'mean',
        'Hour': 'mean'
    }).round(2)
    cluster_summary['Pattern'] = [f"Pattern {i+1}" for i in range(optimal_k)]
    
    st.markdown("**Cluster Characteristics:**")
    st.dataframe(cluster_summary)

# ==================== PREDICTIVE MODELING ====================
st.markdown('<p class="sub-header">🤖 Advanced Predictive Analytics</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Crime Forecasting", "🎯 Case Resolution Prediction", "📊 Model Performance"])

with tab1:
    st.markdown("**Gradient Boosting Regressor for Crime Count Prediction**")
    
    # Prepare time series features
    monthly_data = filtered_df.groupby(['Year', 'Month']).agg({
        'Report Number': 'count',
        'Crime Severity': lambda x: (x == 'High').sum(),
        'Police Deployed': 'mean'
    }).reset_index()
    monthly_data.columns = ['Year', 'Month', 'Crime Count', 'High Severity Count', 'Avg Police']
    
    # Feature engineering for time series
    monthly_data['Month_Sin'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
    monthly_data['Month_Cos'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)
    monthly_data['Lag_1'] = monthly_data['Crime Count'].shift(1)
    monthly_data['Lag_2'] = monthly_data['Crime Count'].shift(2)
    monthly_data['Rolling_Mean_3'] = monthly_data['Crime Count'].rolling(window=3).mean()
    monthly_data = monthly_data.dropna()
    
    if len(monthly_data) > 12:
        # Train model
        X = monthly_data[['Year', 'Month', 'Month_Sin', 'Month_Cos', 'High Severity Count', 
                         'Avg Police', 'Lag_1', 'Lag_2', 'Rolling_Mean_3']]
        y = monthly_data['Crime Count']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        model_gbr.fit(X_train, y_train)
        y_pred = model_gbr.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.3f}")
        col4.metric("MAPE", f"{mape:.2f}%")
        
        # Visualization
        results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
        results_df['Index'] = range(len(results_df))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df['Index'], y=results_df['Actual'], 
                                mode='lines+markers', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=results_df['Index'], y=results_df['Predicted'],
                                mode='lines+markers', name='Predicted', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Actual vs Predicted Crime Counts', xaxis_title='Time Period', yaxis_title='Crime Count')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model_gbr.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance in Crime Prediction')
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== FUTURE CRIME FORECASTING ====================

st.markdown("### 🔮 Future Crime Forecasting")

# Sidebar control for forecasting horizon
forecast_years = st.slider(
    "Select number of future years to forecast",
    min_value=1,
    max_value=5,
    value=2
)

forecast_months = forecast_years * 12

# Use last known data point
last_row = monthly_data.iloc[-1].copy()

future_predictions = []

lag_1 = last_row['Crime Count']
lag_2 = monthly_data.iloc[-2]['Crime Count']
rolling_mean_3 = monthly_data.iloc[-3:]['Crime Count'].mean()

current_year = int(last_row['Year'])
current_month = int(last_row['Month'])

for step in range(1, forecast_months + 1):
    # Increment month/year
    current_month += 1
    if current_month > 12:
        current_month = 1
        current_year += 1

    # Cyclical encoding
    month_sin = np.sin(2 * np.pi * current_month / 12)
    month_cos = np.cos(2 * np.pi * current_month / 12)

    # Prepare feature vector
    X_future = pd.DataFrame([{
        'Year': current_year,
        'Month': current_month,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos,
        'High Severity Count': last_row['High Severity Count'],
        'Avg Police': last_row['Avg Police'],
        'Lag_1': lag_1,
        'Lag_2': lag_2,
        'Rolling_Mean_3': rolling_mean_3
    }])

    # Predict future crime count
    future_crime = model_gbr.predict(X_future)[0]

    future_predictions.append({
        'Year': current_year,
        'Month': current_month,
        'Predicted Crime Count': round(future_crime, 2)
    })

    # Update lags
    lag_2 = lag_1
    lag_1 = future_crime
    rolling_mean_3 = (rolling_mean_3 * 2 + future_crime) / 3

# Create future dataframe
future_df = pd.DataFrame(future_predictions)
future_df['Date'] = pd.to_datetime(
    future_df[['Year', 'Month']].assign(day=1)
)

# Historical data for plotting
historical_df = monthly_data.copy()
historical_df['Date'] = pd.to_datetime(
    historical_df[['Year', 'Month']].assign(day=1)
)

# ==================== VISUALIZATION ====================

fig = go.Figure()

# Historical trend
fig.add_trace(go.Scatter(
    x=historical_df['Date'],
    y=historical_df['Crime Count'],
    mode='lines+markers',
    name='Historical Crime Count',
    line=dict(color='blue')
))

# Future forecast
fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=future_df['Predicted Crime Count'],
    mode='lines+markers',
    name='Forecasted Crime Count',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title=f"Crime Forecast for Next {forecast_years} Year(s)",
    xaxis_title="Date",
    yaxis_title="Crime Count",
    legend_title="Legend"
)

st.plotly_chart(fig, use_container_width=True)

# ==================== FORECAST TABLE ====================

st.markdown("### 📋 Forecasted Crime Counts")
st.dataframe(future_df)


with tab2:
    st.markdown("**Random Forest Classifier for Case Resolution Prediction**")
    
    # Prepare classification data
    classification_df = filtered_df[['Crime Severity', 'Weapon Used', 'Victim Age', 
                                     'Police Deployed', 'Hour', 'DayOfWeek', 'Case Closed Binary']].dropna()
    
    if len(classification_df) > 50:
        # Encode categorical variables
        le_severity = LabelEncoder()
        le_weapon = LabelEncoder()
        
        classification_df['Severity_Encoded'] = le_severity.fit_transform(classification_df['Crime Severity'])
        classification_df['Weapon_Encoded'] = le_weapon.fit_transform(classification_df['Weapon Used'])
        
        X_clf = classification_df[['Severity_Encoded', 'Weapon_Encoded', 'Victim Age', 
                                   'Police Deployed', 'Hour', 'DayOfWeek']]
        y_clf = classification_df['Case Closed Binary']
        
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        rf_model.fit(X_train_clf, y_train_clf)
        y_pred_clf = rf_model.predict(X_test_clf)
        y_pred_proba = rf_model.predict_proba(X_test_clf)[:, 1]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{rf_model.score(X_test_clf, y_test_clf):.3f}")
        col2.metric("AUC-ROC", f"{roc_auc_score(y_test_clf, y_pred_proba):.3f}")
        report = classification_report(y_test_clf, y_pred_clf, output_dict=True)
        col3.metric("Precision (Closed)", f"{report['1']['precision']:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_clf, y_pred_clf)
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Not Closed', 'Closed'], y=['Not Closed', 'Closed'],
                       title='Confusion Matrix - Case Resolution Prediction')
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.markdown("**Detailed Classification Report:**")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn'))

with tab3:
    st.markdown("**Model Performance Comparison & Validation**")
    
    if len(monthly_data) > 12 and len(classification_df) > 50:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Regression Model (GBR) - Metrics**")
            metrics_data = {
                'Metric': ['MAE', 'RMSE', 'R² Score', 'MAPE'],
                'Value': [mae, rmse, r2, mape]
            }
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(metrics_df, x='Metric', y='Value', title='GBR Performance Metrics',
                        color='Value', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Classification Model (RF) - Performance**")
            acc = rf_model.score(X_test_clf, y_test_clf)
            auc = roc_auc_score(y_test_clf, y_pred_proba)
            precision = report['1']['precision']
            recall = report['1']['recall']
            f1 = report['1']['f1-score']
            
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                'Score': [acc, precision, recall, f1, auc]
            })
            
            fig = px.bar(metrics_df, x='Metric', y='Score', 
                        title='Classification Performance Metrics',
                        color='Score', color_continuous_scale='Greens')
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

# ==================== STATISTICAL TESTING ====================
st.markdown('<p class="sub-header">📊 Statistical Hypothesis Testing</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Chi-Square Test: Crime Severity vs Case Closure**")
    contingency_table = pd.crosstab(filtered_df['Crime Severity'], filtered_df['Case Closed'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    st.metric("Chi-Square Statistic", f"{chi2:.2f}")
    st.metric("P-Value", f"{p_value:.4f}")
    
    if p_value < 0.05:
        st.success("✅ Statistically significant relationship (p < 0.05)")
    else:
        st.info("ℹ️ No statistically significant relationship (p ≥ 0.05)")

with col2:
    st.markdown("**T-Test: Police Deployment - Closed vs Open Cases**")
    closed_cases = filtered_df[filtered_df['Case Closed Binary'] == 1]['Police Deployed']
    open_cases = filtered_df[filtered_df['Case Closed Binary'] == 0]['Police Deployed']
    
    t_stat, t_p_value = stats.ttest_ind(closed_cases.dropna(), open_cases.dropna())
    
    st.metric("T-Statistic", f"{t_stat:.2f}")
    st.metric("P-Value", f"{t_p_value:.4f}")
    
    if t_p_value < 0.05:
        st.success(f"✅ Significant difference in deployment (p < 0.05)")
        st.info(f"Mean police deployment: Closed={closed_cases.mean():.1f}, Open={open_cases.mean():.1f}")
    else:
        st.info("ℹ️ No significant difference in deployment")

# ==================== ACTIONABLE RECOMMENDATIONS ====================
st.markdown('<p class="sub-header">💡 AI-Powered Recommendations & Insights</p>', unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-box">
<h4>🎯 Strategic Recommendations Based on Advanced Analytics:</h4>
<ol>
<li><b>Resource Allocation:</b> Deploy additional {int(avg_police * 0.2)} officers during peak hours ({peak_hour}:00-{peak_hour+2}:00) in high-risk cities</li>
<li><b>High-Risk Zones:</b> {highest_risk['City']} requires immediate attention with Risk Score of {highest_risk['Risk Score']:.1f} - recommend 15-20% increase in police deployment</li>
<li><b>Prevention Strategy:</b> Focus preventive measures during identified peak crime hours and weekend periods</li>
<li><b>Case Management:</b> Fast-track high-severity crimes - ML model predicts {report['1']['precision']*100:.1f}% precision in identifying closeable cases</li>
<li><b>Predictive Policing:</b> Use GBR model (R²={r2:.3f}) for monthly crime forecasting and proactive resource planning</li>
<li><b>Cluster-Based Interventions:</b> Implement targeted strategies for each of the {optimal_k} identified crime patterns</li>
</ol>
</div>
""", unsafe_allow_html=True)

# ==================== DOWNLOAD SECTION ====================
st.markdown('<p class="sub-header">📥 Export Analysis Results</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="⬇️ Download Filtered Dataset (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_crime_data.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="⬇️ Download City Risk Scores",
        data=risk_scores.to_csv(index=False),
        file_name="city_risk_scores.csv",
        mime="text/csv"
    )

with col3:
    if 'feature_importance' in locals():
        st.download_button(
            label="⬇️ Download Feature Importance",
            data=feature_importance.to_csv(index=False),
            file_name="feature_importance.csv",
            mime="text/csv"
        )

# ==================== FOOTER ====================
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:0.9rem; color:gray;">
        🚨 <b>Advanced Crime Analytics & Risk Management System</b><br>
        Built using Streamlit · Scikit-learn · Plotly · Pandas<br>
        For academic & research purposes only
    </div>
    """,
    unsafe_allow_html=True
)
