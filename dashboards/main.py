import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# Configure page
st.set_page_config(
    page_title="F1 Race Winner Predictions",
    page_icon="🏎️",
    layout="wide"
)

# Title and description
st.title("🏎️ F1 Race Winner Predictions Dashboard")
st.markdown("""
This dashboard provides real-time insights into F1 race winner predictions,
model performance metrics, and historical accuracy.
""")

# Sidebar
st.sidebar.header("Configuration")
selected_race = st.sidebar.selectbox(
    "Select Race",
    ["Monaco GP", "Silverstone", "Spa", "Monza", "Suzuka"]
)

# Load latest predictions
@st.cache_data(ttl=300)
def load_latest_predictions():
    try:
        response = requests.get("http://localhost:8000/predictions/latest")
        return response.json()
    except:
        return None

# Load model performance metrics
@st.cache_data(ttl=3600)
def load_model_metrics():
    try:
        metrics_path = Path("models/metrics.json")
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
        return None
    except:
        return None

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest Predictions")
    predictions = load_latest_predictions()
    
    if predictions:
        # Create prediction visualization
        fig = go.Figure(data=[
            go.Bar(
                x=list(predictions.keys()),
                y=list(predictions.values()),
                text=list(predictions.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Driver Win Probabilities",
            xaxis_title="Driver",
            yaxis_title="Probability",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No predictions available")

with col2:
    st.subheader("Model Performance")
    metrics = load_model_metrics()
    
    if metrics:
        # Create metrics visualization
        fig = go.Figure(data=[
            go.Indicator(
                mode="gauge+number",
                value=metrics['accuracy'],
                title={'text': "Model Accuracy"},
                gauge={'axis': {'range': [0, 1]}}
            )
        ])
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metrics available")

# Historical Performance
st.subheader("Historical Performance")
col3, col4 = st.columns(2)

with col3:
    # Load historical data
    try:
        historical_data = pd.read_parquet("data/processed/historical_predictions.parquet")
        
        # Create time series plot
        fig = px.line(
            historical_data,
            x='date',
            y='accuracy',
            title='Prediction Accuracy Over Time'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("No historical data available")

with col4:
    # Feature importance
    try:
        feature_importance = pd.read_parquet("models/feature_importance.parquet")
        
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("No feature importance data available")

# Model Monitoring
st.subheader("Model Monitoring")
col5, col6 = st.columns(2)

with col5:
    # Data drift
    try:
        drift_data = pd.read_parquet("monitoring/data_drift.parquet")
        
        fig = px.line(
            drift_data,
            x='date',
            y='drift_score',
            title='Data Drift Over Time'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("No drift data available")

with col6:
    # Prediction distribution
    try:
        pred_dist = pd.read_parquet("monitoring/prediction_distribution.parquet")
        
        fig = px.histogram(
            pred_dist,
            x='predicted_position',
            title='Prediction Distribution'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("No prediction distribution data available")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Last updated: {}</p>
    <p>Data source: F1 API, FastF1, OpenWeatherMap</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True) 