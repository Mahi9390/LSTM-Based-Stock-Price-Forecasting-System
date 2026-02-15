# app.py  (for Streamlit deployment)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          # Suppress TensorFlow logs
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
MODEL_PATH          = "optimal_lstm_model.h5"
SCALER_PATH         = "scaler.pkl"
LOOK_BACK           = 60
DEFAULT_PREDICTION_DAYS = 30

# ────────────────────────────────────────────────
# Load model & scaler (done once when app starts)
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_resources()

# ────────────────────────────────────────────────
# Data fetching
# ────────────────────────────────────────────────
def fetch_historical_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None, "No data found for this ticker and date range."

        # Handle possible MultiIndex / weird columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = df.columns.str.lower()

        if "close" not in df.columns:
            return None, "Close price column not found."

        df = df[["close"]].rename(columns={"close": "Close"})
        df.index.name = "Date"
        df = df.sort_index()
        return df, None

    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# ────────────────────────────────────────────────
# Forecasting logic
# ────────────────────────────────────────────────
def generate_forecast(historical_df, days):
    if len(historical_df) < LOOK_BACK:
        return None, f"Need at least {LOOK_BACK} days of data. Got {len(historical_df)}."

    close_values   = historical_df["Close"].values.reshape(-1, 1)
    scaled_close   = scaler.transform(close_values)
    current_batch  = scaled_close[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)

    predictions_scaled = []
    for _ in range(days):
        pred = model.predict(current_batch, verbose=0)[0, 0]
        predictions_scaled.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()

    future_dates = pd.date_range(
        start = historical_df.index[-1] + timedelta(days=1),
        periods = days
    )

    return future_dates, predictions

# ────────────────────────────────────────────────
# Plot generation → returns base64 image
# ────────────────────────────────────────────────
def generate_plot(historical_df, future_dates, predictions, ticker):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Show last 200 days of history (or all if less)
    hist_window = historical_df[-200:]

    ax.plot(hist_window.index, hist_window["Close"],
            label="Historical Close", color="#00ffcc", linewidth=2.1)

    ax.plot(future_dates, predictions,
            label=f"Forecast ({len(predictions)} days)",
            color="#ff6b6b", linestyle="--", linewidth=2.4, marker='o', markersize=4)

    ax.set_title(f"{ticker.upper()} Stock Price Forecast (LSTM)", fontsize=15, color="white")
    ax.set_xlabel("Date", fontsize=11, color="white")
    ax.set_ylabel("Price ($)", fontsize=11, color="white")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25, color="gray")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches='tight', facecolor='#0e1117')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return f"data:image/png;base64,{plot_url}"

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.set_page_config(page_title="LSTM Stock Price Forecaster", layout="wide")

st.title("LSTM-Based Stock Price Forecasting")
st.markdown("Enter a stock ticker and forecast future prices using a trained LSTM model.")

# ── Sidebar controls ───────────────────────────────────────
with st.sidebar:
    st.header("Forecast Settings")

    ticker = st.text_input("Stock Ticker", value="AAPL").upper().strip()

    today = datetime.today()
    default_start = today - timedelta(days=5*365)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start)
    with col2:
        end_date = st.date_input("End Date", value=today)

    prediction_days = st.slider(
        "Days to Forecast",
        min_value=1,
        max_value=365,
        value=DEFAULT_PREDICTION_DAYS,
        step=1
    )

    forecast_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

# ── Main area ──────────────────────────────────────────────
if forecast_btn and ticker:

    with st.spinner("Fetching historical data..."):
        df, error = fetch_historical_data(ticker, start_date, end_date)

    if error:
        st.error(error)
    else:
        with st.spinner("Generating forecast..."):
            future_dates, predictions = generate_forecast(df, prediction_days)

            if isinstance(predictions, str):  # error case
                st.error(predictions)
            else:
                # Show plot
                plot_url = generate_plot(df, future_dates, predictions, ticker)
                st.image(plot_url, use_column_width=True)

                # Show forecast table
                forecast_df = pd.DataFrame({
                    "Date": [d.date() for d in future_dates],
                    "Predicted Close": [f"${p:,.2f}" for p in predictions]
                })

                st.subheader(f"Forecast for {ticker} – Next {prediction_days} days")
                st.dataframe(
                    forecast_df.style.format({"Predicted Close": "${:,.2f}"}),
                    use_container_width=True,
                    hide_index=True
                )

                # Optional: raw numbers chart
                st.subheader("Forecast Chart (alternative view)")
                chart_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted": predictions
                }).set_index("Date")
                st.line_chart(chart_df)

else:
    st.info("Enter a ticker and click 'Generate Forecast' to begin.")

st.markdown("---")
st.caption("Model • optimal_lstm_model.h5  •  Look-back window: 60 days")
