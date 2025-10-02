
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
import pmdarima as pm

st.title("Sales Prediction and Visualization Tool")

# =========================
# File Upload Section
# =========================
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.write("Preview of uploaded data:")
    st.write(df.head())

    # Validate columns
    if "Date" not in df.columns or "Sales" not in df.columns:
        st.error("Data must contain 'Date' and 'Sales' columns")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # =========================
    # Visualization BEFORE prediction
    # =========================
    st.subheader("Visualize Your Uploaded Data")
    viz_choice = st.selectbox(
        "Choose a visualization type for uploaded data",
        ["Line Chart", "Bar Chart", "Histogram", "Scatter Plot"]
    )

    if viz_choice == "Line Chart":
        st.line_chart(df.set_index("Date")["Sales"])
    elif viz_choice == "Bar Chart":
        st.bar_chart(df.set_index("Date")["Sales"])
    elif viz_choice == "Histogram":
        bins = st.slider("Number of bins", 5, 50, 10)
        fig, ax = plt.subplots()
        ax.hist(df["Sales"], bins=bins)
        st.pyplot(fig)
    elif viz_choice == "Scatter Plot":
        fig, ax = plt.subplots()
        ax.scatter(df["Date"], df["Sales"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        st.pyplot(fig)

    # =========================
    # Forecasting Section
    # =========================
    st.subheader("Forecast Sales")
    model_choice = st.selectbox("Choose forecasting model", ["Prophet", "ARIMA"])
    forecast_periods = st.number_input("Forecast months into future", min_value=1, max_value=36, value=12)

    forecast_df = pd.DataFrame()

    if model_choice == "Prophet":
        prophet_df = df.rename(columns={"Date": "ds", "Sales": "y"})
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=forecast_periods, freq="M")
        forecast = m.predict(future)
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_periods)
        forecast_df = forecast_df.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"})

    elif model_choice == "ARIMA":
        series = df.set_index("Date")["Sales"]
        model = pm.auto_arima(series, seasonal=True, m=12, trace=False, error_action="ignore", suppress_warnings=True)
        forecast, conf_int = model.predict(n_periods=forecast_periods, return_conf_int=True)
        future_dates = pd.date_range(df["Date"].iloc[-1] + pd.offsets.MonthBegin(), periods=forecast_periods, freq="M")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast,
            "Lower": conf_int[:, 0],
            "Upper": conf_int[:, 1]
        })

    st.write("Forecasted Sales Preview:")
    st.write(forecast_df)

    # =========================
    # Visualization AFTER prediction
    # =========================
    st.subheader("Visualize Forecast Results")
    forecast_viz_choice = st.selectbox(
        "Choose how you want to visualize forecast results",
        ["Line Chart with Forecast", "Table", "Bar Chart", "Confidence Interval Plot"]
    )

    if forecast_viz_choice == "Line Chart with Forecast":
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Sales"], label="Historical")
        ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", linestyle="dashed")
        ax.legend()
        st.pyplot(fig)

    elif forecast_viz_choice == "Table":
        st.write(forecast_df)

    elif forecast_viz_choice == "Bar Chart":
        fig, ax = plt.subplots()
        ax.bar(forecast_df["Date"], forecast_df["Forecast"])
        st.pyplot(fig)

    elif forecast_viz_choice == "Confidence Interval Plot":
        if "Lower" in forecast_df.columns and "Upper" in forecast_df.columns:
            fig, ax = plt.subplots()
            ax.plot(df["Date"], df["Sales"], label="Historical")
            ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", linestyle="dashed")
            ax.fill_between(forecast_df["Date"], forecast_df["Lower"], forecast_df["Upper"], color="gray", alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Confidence intervals not available for this model.")
