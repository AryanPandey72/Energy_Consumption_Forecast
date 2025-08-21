import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import tensorflow as tf
from tensorflow.keras import models
from io import BytesIO

# Import your existing functions
from your_module import (  # replace 'your_module' with the filename of your project script without .py
    load_and_prepare_household_power,
    plot_eda,
    train_and_evaluate,
    DATA_TXT_PATH,
    TARGET_COL
)

st.set_page_config(page_title="⚡ Energy Consumption Forecasting", layout="wide")

st.title("⚡ Energy Consumption Forecasting using LSTM")

# File upload
uploaded_file = st.file_uploader("Upload household_power_consumption.txt", type=["txt"])
if uploaded_file is not None:
    with open("uploaded_data.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    data_path = "uploaded_data.txt"
else:
    data_path = DATA_TXT_PATH  # fallback default

if st.button("Run Forecasting"):
    try:
        with st.spinner("Loading and preparing dataset..."):
            df = load_and_prepare_household_power(data_path)

        st.subheader("Exploratory Data Analysis")
        # EDA plots
        plot_eda(df)
        st.image("reports/figures/eda_target.png", caption="Target Series")
        st.image("reports/figures/eda_hourly_mean.png", caption="Hourly Mean")
        st.image("reports/figures/eda_weekly_mean.png", caption="Weekly Mean")

        with st.spinner("Training LSTM model... This may take a few minutes ⏳"):
            results = train_and_evaluate(df)

        st.subheader("📊 Evaluation Metrics")
        metrics = results["metrics"]
        st.json(metrics)

        st.subheader("📈 Forecast Results")
        forecast_df = results["forecast_df"]
        st.line_chart(forecast_df)

        # Download forecast
        csv = forecast_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast_latest.csv",
            mime="text/csv"
        )

        # Show saved plots
        st.subheader("Training and Forecast Plots")
        st.image("reports/figures/learning_curves.png", caption="Learning Curves")
        st.image("reports/figures/horizon_errors.png", caption="Horizon Errors")
        st.image("reports/figures/residuals_hist.png", caption="Residuals Histogram")
        st.image("reports/figures/sample_forecast.png", caption="Sample Forecast")
        st.image("reports/figures/history_forecast_overlay.png", caption="History vs Forecast")

        st.success("✅ Forecasting complete. Artifacts saved in 'reports/'.")

    except Exception as e:
        st.error(f"Error: {e}")
