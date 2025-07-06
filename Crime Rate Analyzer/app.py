import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import streamlit as st  # type: ignore
import plotly.express as px  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore

# Set the title of the app
st.title('Crime Rate Analysis in India')

# Load the data
@st.cache_data  # Updated caching method
def load_data():
    df = pd.read_csv('crime_dataset_india.csv')
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_city = st.sidebar.selectbox("Select a City", ["All"] + list(df["City"].dropna().unique()))
selected_crime = st.sidebar.selectbox("Select Crime Type", ["All"] + list(df["Crime Description"].dropna().unique()))

# Apply filters
if selected_city != "All":
    df = df[df["City"] == selected_city]
if selected_crime != "All":
    df = df[df["Crime Description"] == selected_crime]

# Display the first few rows of the dataset
st.subheader('Dataset Preview')
st.write(df.head())

# Check for missing values
st.subheader('Missing Values')
missing_values = df.isnull().sum()
st.write(missing_values[missing_values > 0])

# Fill missing values with NaN for columns with missing values
for column in df.columns[df.isnull().any()]:
    df[column].fillna(np.nan, inplace=True)

# Convert date columns to datetime
df['Date Reported'] = pd.to_datetime(df['Date Reported'], errors='coerce', dayfirst=True)
df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], errors='coerce', dayfirst=True)
df['Date Case Closed'] = pd.to_datetime(df['Date Case Closed'], errors='coerce', dayfirst=True)

df['Year'] = df['Date Reported'].dt.year

# Encode categorical columns
df['Victim Gender'] = df['Victim Gender'].astype('category')
df['Weapon Used'] = df['Weapon Used'].astype('category')
df['Crime Domain'] = df['Crime Domain'].astype('category')

# Calculate time to close case (in days)
df['Time to Close Case'] = (df['Date Case Closed'] - df['Date Reported']).dt.days

# Summary statistics for numerical columns
st.subheader('Summary Statistics for Numerical Columns')
st.write(df.describe())

# Summary statistics for categorical columns
st.subheader('Summary Statistics for Categorical Columns')
st.write(df.describe(include=['category']))

# Crime Distribution by City
st.subheader('Crime Distribution by City')
city_crime_count = df['City'].value_counts().reset_index()
city_crime_count.columns = ['City', 'Crime Count']
fig = px.bar(city_crime_count, x='City', y='Crime Count', title='Crime Distribution by City', color='Crime Count', color_continuous_scale='viridis')
st.plotly_chart(fig, use_container_width=True)

# Crime Distribution by Type
st.subheader('Crime Distribution by Type')
crime_type_count = df['Crime Description'].value_counts().reset_index()
crime_type_count.columns = ['Crime Type', 'Crime Count']
fig = px.bar(crime_type_count, x='Crime Type', y='Crime Count', title='Crime Distribution by Type', color='Crime Count', color_continuous_scale='magma')
st.plotly_chart(fig, use_container_width=True)

# Victim Age Distribution
st.subheader('Victim Age Distribution')
fig = px.histogram(df, x='Victim Age', nbins=20, title='Victim Age Distribution', color_discrete_sequence=['blue'])
st.plotly_chart(fig, use_container_width=True)

# Weapon Used in Crimes
st.subheader('Weapon Used in Crimes')
weapon_count = df['Weapon Used'].value_counts().reset_index()
weapon_count.columns = ['Weapon', 'Crime Count']
fig = px.bar(weapon_count, x='Weapon', y='Crime Count', title='Weapon Used in Crimes', color='Crime Count', color_continuous_scale='plasma')
st.plotly_chart(fig, use_container_width=True)

# Crime Trends Over Time
st.subheader('Crime Trends Over Time')
crime_trend = df.groupby('Year').size().reset_index(name='Crime Count')
fig = px.line(crime_trend, x='Year', y='Crime Count', title='Crime Trends Over the Years', markers=True)
st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.subheader('Crime Data Correlation Heatmap')
numerical_cols = df.select_dtypes(include=['number']).columns
corr_matrix = df[numerical_cols].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Predict future crime rates
st.subheader('Crime Rate Prediction')

# Prepare data for model
X = crime_trend[['Year']]
y = crime_trend['Crime Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display model metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")

# Predict future values
future_years = pd.DataFrame({'Year': range(df['Year'].max() + 1, df['Year'].max() + 6)})
future_predictions = model.predict(future_years)
future_df = pd.DataFrame({'Year': future_years['Year'], 'Predicted Crime Count': future_predictions})

# Display predictions
st.subheader('Future Crime Rate Predictions')
st.write(future_df)

# Plot predictions
fig = px.line(future_df, x='Year', y='Predicted Crime Count', title='Predicted Crime Trends', markers=True)
st.plotly_chart(fig, use_container_width=True)

st.write("### End of Analysis")
