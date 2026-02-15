DEMO LINK:https://lstm-based-stock-price-forecasting-system-g2cb9s3thbxxujhup89d.streamlit.app/

📈 LSTM-Based Stock Price Forecasting System
📌 Project Overview

This project implements an end-to-end time series forecasting system to predict future stock closing prices using classical statistical models, machine learning, and deep learning techniques. The system performs extensive EDA, feature engineering, model comparison, and deploys the best-performing LSTM model as a web application.

Both Streamlit and Flask are used for deployment, demonstrating flexibility in production-ready ML application development.

🔍 Data Collection

Stock market data fetched dynamically using Yahoo Finance (yfinance)

Historical data range: 2012 – 2019

Example stock: AAPL (Apple Inc.)

📊 Exploratory Data Analysis (EDA)

Time-series visualization of closing prices

Seasonal decomposition (trend, seasonality, residuals)

Stationarity testing using ADF Test

Autocorrelation & Partial Autocorrelation analysis (ACF, PACF)

Outlier detection using IQR

Monthly and quarterly resampling analysis

🛠️ Feature Engineering

Lag features (1, 2, 3, 5, 7, 14, 21 days)

Rolling statistics (mean, volatility)

Returns (percentage & log returns)

Technical indicators:

Moving Averages (MA 7, 14, 30)

RSI (14)

MACD & Signal Line

Bollinger Bands

EMA (9, 21, 50)

Date-based features (day of week, month-end)

🤖 Models Implemented

The following models were trained and evaluated using RMSE & MAE:

Model	Description
ARIMA	Classical time series forecasting
SARIMA	Seasonal ARIMA
XGBoost	Tree-based regression with engineered features
LSTM	Deep learning model for sequential data
Prophet	Facebook Prophet for trend & seasonality
🏆 Model Evaluation & Selection

All models were evaluated on an 80/20 time-based split

LSTM achieved the lowest RMSE and MAE

Final model selected based on performance comparison

🔧 LSTM Hyperparameter Tuning

Grid search was performed over:

LSTM units: 50, 75, 100

Number of layers: 1, 2

Epochs: 20, 30

Batch sizes: 32, 64

The best-performing configuration was retrained on the full dataset and saved for deployment.

🔮 30-Day Forecasting

Uses the last 60 time steps as input

Predicts the next 30 trading days

Results visualized alongside historical prices

🚀 Deployment
✅ Streamlit Deployment

Interactive UI with:

Stock ticker selection

Forecast horizon input

Dynamic Plotly charts

Data tables for predictions

Deployed publicly using ngrok

✅ Flask Deployment

Flask backend used for:

Model loading

Forecast generation

Plot rendering

Deployed locally via VS Code

Demonstrates production-style REST deployment

🧪 Tech Stack

Languages: Python

Libraries: pandas, numpy, matplotlib, seaborn, statsmodels

ML/DL: scikit-learn, XGBoost, TensorFlow/Keras, Prophet

Deployment: Streamlit, Flask, ngrok

Visualization: Matplotlib, Plotly

📂 Project Structure
├── data/
├── notebooks/
│   ├── EDA & Feature Engineering.ipynb
│   ├── Model Comparison.ipynb
│   ├── LSTM Hyperparameter Tuning.ipynb
├── app.py                # Streamlit app
├── flask_app.py          # Flask deployment
├── optimal_lstm_model.h5
├── scaler.pkl
├── README.md

📈 Key Outcomes

Built a complete time-series ML pipeline

Compared classical, ML, and deep learning models

Achieved best performance using LSTM

Successfully deployed the model using both Streamlit and Flask

Demonstrated real-world, end-to-end ML engineering skills

👤 Author

Mahesh
Aspiring Data Scientist | Time Series | Machine Learning | Deep Learning
