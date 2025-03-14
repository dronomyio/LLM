import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.title('Stock and Bond Analysis Dashboard')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Stock_Bond.csv')
    return df

# Display a loading message while data is being processed
with st.spinner('Loading data...'):
    df = load_data()

# Show dataset info
st.subheader('Dataset Overview')
st.write(f"Shape: {df.shape}")

# Show the first few rows
st.subheader('Sample Data')
st.dataframe(df.head())

# Data statistics
st.subheader('Statistical Summary')
st.dataframe(df.describe())

# Time series plot
st.subheader('Time Series Analysis')
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Select columns for plotting
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_column = st.selectbox('Select column for time series', numeric_columns)
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df[selected_column])
    ax.set_xlabel('Date')
    ax.set_ylabel(selected_column)
    ax.set_title(f'{selected_column} Over Time')
    st.pyplot(fig)

# Correlation heatmap
st.subheader('Correlation Analysis')
numeric_data = df.select_dtypes(include=['float64', 'int64'])
if not numeric_data.empty:
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = numeric_data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# Histogram
st.subheader('Distribution Analysis')
histogram_column = st.selectbox('Select column for histogram', 
                              df.select_dtypes(include=['float64', 'int64']).columns)
fig, ax = plt.subplots()
sns.histplot(df[histogram_column], kde=True, ax=ax)
st.pyplot(fig)

# Scatter plot
st.subheader('Relationship Analysis')
col1 = st.selectbox('Select X-axis', df.select_dtypes(include=['float64', 'int64']).columns, key='x')
col2 = st.selectbox('Select Y-axis', df.select_dtypes(include=['float64', 'int64']).columns, key='y')
fig, ax = plt.subplots()
sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
st.pyplot(fig)