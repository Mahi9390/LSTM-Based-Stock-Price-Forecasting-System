
from flask import Flask, render_template, request
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

warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
MODEL_PATH = "optimal_lstm_model.h5"
SCALER_PATH = "scaler.pkl"
LOOK_BACK = 60
DEFAULT_PREDICTION_DAYS = 30

app = Flask(__name__)

# ---------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------------------------------
# Data Fetching (Improved Handling)
# ---------------------------------------------------
def fetch_historical_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None, "No data found for the given ticker and date range."

        df = df.reset_index()

        # Robust column flattening
        if isinstance(df.columns, pd.MultiIndex):
            # Take the first level (field names like Open, High, etc.)
            df.columns = df.columns.get_level_values(0)
        # In case it's already flat but has tuples somehow
        elif any(isinstance(c, tuple) for c in df.columns):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        df.columns = df.columns.str.lower()

        if "close" not in df.columns:
            return None, "Close price data not available."

        df.set_index("date", inplace=True)
        df.rename(columns={"close": "Close"}, inplace=True)
        df = df.sort_index()

        return df, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# ---------------------------------------------------
# Forecast Logic
# ---------------------------------------------------
def generate_forecast(historical_df, days):
    if len(historical_df) < LOOK_BACK:
        return None, f"Not enough data (need at least {LOOK_BACK} days)"

    close_values = historical_df["Close"].values.reshape(-1, 1)
    scaled_close = scaler.transform(close_values)

    current_batch = scaled_close[-LOOK_BACK:].reshape((1, LOOK_BACK, 1))
    predictions_scaled = []

    for _ in range(days):
        pred = model.predict(current_batch, verbose=0)[0, 0]  # [0] since it's [[value]]
        predictions_scaled.append(pred)
        # Shift and append new prediction
        current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=historical_df.index[-1] + timedelta(days=1), periods=days)

    return future_dates, predictions

# ---------------------------------------------------
# Plot Generation
# ---------------------------------------------------
def generate_plot(historical_df, future_dates, predictions):
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))

    # Show last 200 days for better context
    hist_window = historical_df[-200:]
    plt.plot(hist_window.index, hist_window["Close"], label="Historical Close", color="#00ffcc", linewidth=2)

    plt.plot(future_dates, predictions, label=f"Forecast ({len(predictions)} days)",
             color="#ff6b6b", linestyle="--", linewidth=2.5, marker='o', markersize=4)

    plt.title(f"{ticker.upper()} Stock Price Forecast using LSTM", fontsize=16, color="white")
    plt.xlabel("Date", fontsize=12, color="white")
    plt.ylabel("Price ($)", fontsize=12, color="white")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=150, bbox_inches='tight', facecolor='#1e3c72')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return plot_url

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global ticker  # To use in plot title
    plot_url = None
    forecast_df = None
    error = None
    ticker = "AAPL"
    today = datetime.today()
    start_date = (today - timedelta(days=5*365)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    prediction_days = DEFAULT_PREDICTION_DAYS

    if request.method == "POST":
        ticker = request.form.get("ticker", "AAPL").upper().strip()
        start_date = request.form.get("start_date", start_date)
        end_date = request.form.get("end_date", end_date)
        
        try:
            prediction_days = int(request.form.get("prediction_days", DEFAULT_PREDICTION_DAYS))
            if not (1 <= prediction_days <= 365):
                raise ValueError
        except:
            error = "Prediction days must be a number between 1 and 365."
            prediction_days = DEFAULT_PREDICTION_DAYS

        if not error:
            historical_df, fetch_error = fetch_historical_data(ticker, start_date, end_date)
            if fetch_error:
                error = fetch_error
            else:
                future_dates, predictions = generate_forecast(historical_df, prediction_days)
                if isinstance(predictions, str):  # error message
                    error = predictions
                else:
                    plot_url = generate_plot(historical_df, future_dates, predictions)

                    forecast_df = pd.DataFrame({
                        "Date": [d.date() for d in future_dates],
                        "Predicted Close Price": [f"${p:.2f}" for p in predictions]
                    })

    return render_template(
        "index.html",
        plot_url=plot_url,
        forecast=forecast_df,
        error=error,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        prediction_days=prediction_days
    )

if __name__ == "__main__":
    app.run(debug=True)