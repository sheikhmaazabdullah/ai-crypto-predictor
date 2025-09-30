import streamlit as st
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import pymongo
import pymssql
import mysql.connector
import json
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
# import keras_tuner as kt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, LayerNormalization, MultiHeadAttention, Flatten
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import asyncio
import os
import platform
import warnings
import traceback
import nest_asyncio
import logging
import plotly.express as px
from fpdf import FPDF
from PIL import Image
import pyttsx3

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
nest_asyncio.apply()

# --- GLOBAL SETTINGS & THEME CUSTOMIZATION ---
st.set_page_config(layout="wide", page_title="AI Crypto Predictor")

st.markdown("""
    <style>
        .reportview-container {
            flex-direction: column;
        }
        .main {
            padding-top: 0;
        }
        .stButton>button {
            width: 100%;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

def toggle_theme():
    st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"

CUSTOM_COLORS = {
    "Dark": {
        "bg": "#1E1E1E",
        "fg": "#E0E0E0",
        "primary": "#007BFF",
        "secondary_bg": "#2D2D2D",
        "grid": "rgba(128, 128, 128, 0.2)"
    },
    "Light": {
        "bg": "#FFFFFF",
        "fg": "#333333",
        "primary": "#1976D2",
        "secondary_bg": "#F0F2F6",
        "grid": "rgba(128, 128, 128, 0.1)"
    }
}
colors = CUSTOM_COLORS[st.session_state.theme]

    # Main content
    tabs = st.tabs(["Documentation"])

    with tabs[10]:
        st.header("📚 Documentation")
        st.markdown(
            """
            # 📘 Crypto & Stock Predictor — Full Documentation
            ## 🔍 Overview
            This Streamlit-based dashboard enables **real-time crypto & stock analysis**, using:
            - Exchange integration (Binance, Kraken, Bybit, MEXC, Bitget, BingX via CCXT)
            - Multiple predictive models (AI/ML, ARIMA, Linear)
            - Advanced technical indicators & strategies
            - Portfolio simulation and visual trading signals
            - Dynamic charts, buy/sell zones, and backtesting
            Ideal for traders, data scientists, and developers.
            ## 🛠️ Technologies Used
            - **Frontend/UI**: Streamlit
            - **Data & Indicators**: `pandas`, `ta` (for technical analysis), `ccxt.async_support` (for exchange integration)
            - **Charts**: Plotly (for interactive visualizations)
            - **Models**: LSTM, GRU, Transformer (via TensorFlow/Keras), Linear Regression (from scikit-learn), ARIMA (from statsmodels)
            - **Database**: MongoDB, MySQL, MSSQL support (for persistence)
            - **Others**: NewsAPI (for sentiment analysis), `fpdf2` (for PDF reports), `Pillow` (for image handling in PDF)
            ## 🚀 Setup & Installation
            1. **Install dependencies**:
               ```bash
               pip install streamlit ccxt pandas numpy plotly ta requests pymongo pymssql mysql-connector-python websockets tensorflow keras-tuner joblib scikit-learn statsmodels matplotlib seaborn fpdf2 Pillow
               ```
               *Note: Ensure you have the correct database drivers installed for MSSQL (`pymssql`) or MySQL (`mysql-connector-python`).*
            2. **Run the app**:
               ```bash
               streamlit run your_app_file_name.py
               ```
            3. **Optional**:
               - **NewsAPI Key**: Obtain your free API key from [https://newsapi.org](https://newsapi.org) to enable sentiment analysis.
               - **Database Credentials**: If you wish to save your preferences, trade history, and news articles persistently, configure your database connection details in the sidebar.
            ## 🧭 Sidebar Settings Guide
            - **Database Settings**:
                - **Database Type**: Choose "None" if you don't want to use a database, or select "MongoDB", "MSSQL", or "MySQL" for persistence.
                - **DB Host, Port, Username, Password, Name**: Enter your database connection details.
                - **User ID**: A unique identifier for saving/loading your preferences and trade history.
                - **Connect & Load/Save**: Click this button to establish a database connection and load/save your settings.
            - **Market Data Settings**:
                - **Select Exchange**: Choose from supported cryptocurrency exchanges (Binance, Kraken, Bybit, MEXC, Bitget, BingX).
                - **Select Symbols**: Choose one or more trading pairs (e.g., BTC/USDT). The list dynamically updates based on the selected exchange.
                - **Select Timeframe**: Choose the candlestick interval (e.g., 1m, 5m, 1h, 1d).
            - **Technical Indicators**:
                - **Checkboxes**: Enable/disable various technical indicators to be displayed on the charts and used in strategies (Moving Averages, Bollinger Bands, RSI, MACD, VWAP, Stochastic, ATR, Momentum).
                - **Indicator Parameters (Expandable)**: Fine-tune the settings for each indicator (e.g., MA periods, RSI thresholds, Bollinger Bands window).
            - **Prediction Models**:
                - **Select Prediction Model**: Choose the AI model for price forecasting (LSTM, Transformer, ARIMA, Linear Regression, Ensemble).
                - **Prediction Horizon (periods)**: The number of future candles the AI model will predict.
                - **Training Epochs**: Number of training iterations for deep learning models.
                - **Hyperparameter Tuning Epochs**: Number of epochs for Keras Tuner to find optimal model parameters.
            - **Trading Strategies**:
                - **Checkboxes**: Enable/disable different trading strategies for simulation and backtesting (Moving Average Crossover, Mean Reversion, Momentum, Bollinger Bands, RSI Divergence, Combined Indicators).
            - **Simulation Settings**:
                - **Initial Investment ($)**: The starting capital for simulated trades.
            - **NewsAPI Key**:
                - Enter your NewsAPI key here to enable sentiment analysis in the application.
            - **Auto-Refresh**:
                - **Auto-refresh interval (seconds, 0 for off)**: Automatically refreshes the dashboard at set intervals.
            ## 🧪 Tabs Breakdown — What Each Tab Does
            ### 📈 **Dashboard Tab**
            - Displays live candlestick charts with selected technical indicators.
            - Shows AI-powered price predictions as a dashed line with a confidence interval.
            - Provides key market metrics: Current Price, Trend, Support & Resistance levels, Funding Rate, Order Book Imbalance, and Sentiment Score.
            - Offers educational insights into common indicators.
            - Allows downloading a comprehensive PDF report of the analysis.
            ### 📊 **Model Performance Tab**
            - Presents performance metrics (RMSE, MAE) for the selected AI model.
            - Visualizes confidence intervals around predictions.
            - Includes a conceptual section for comparing different models.
            - Discusses Explainable AI (XAI) insights (e.g., SHAP values).
            ### 📖 **Order Book Tab**
            - Visualizes the real-time order book depth with cumulative bids and asks.
            - Displays Bid-Ask Spread and Order Book Imbalance, crucial for understanding market pressure.
            ### 🌍 **Multi-Exchange Tab**
            - Compares symbol availability and simulated market metrics (liquidity, volatility, funding rate) across multiple exchanges.
            - Helps in identifying the best exchange for a given trading pair.
            ### 🤖 **Simulation & Strategies Tab**
            - Runs a virtual auto-trading bot simulation based on selected strategies **on predicted candles**.
            - Shows detailed trade logs for each strategy, including entry/exit prices, shares, and profit/loss.
            - Provides an overall PnL dashboard with Total Return, Sharpe Ratio, and Max Drawdown.
            - Plots the simulated equity curve, showing portfolio value over time.
            ### 🔍 **Backtest Tab**
            - Evaluates the historical performance of your chosen trading strategies.
            - Calculates key performance metrics for each strategy: Sharpe Ratio, Total Return, and Max Drawdown.
            - Helps you understand how strategies would have performed in the past.
            ### 📚 **Detailed Predictions Tab**
            - Provides a granular view of the last 5 actual candles and predicted future candles.
            - Lists individual price predictions with percentage change from the last known price.
            - Embeds a live TradingView chart for real-time market visualization.
            ### 📈 **Comparison Tab**
            - Compares saved strategy predictions against actual historical prices.
            - Displays Mean Prediction Error, Mean Error Percentage, and Direction Hit Rate.
            - Visualizes predicted vs. actual price curves with trade markers.
            - Shows a detailed trade comparison table with realized profits.
            ### 💬 **AI Assistant Tab**
            - A conversational AI that can answer questions about market data, predictions, and indicators.
            - Now includes Text-to-Speech (TTS) for spoken responses.
            ### 📰 **News Tab**
            - Fetches and displays the latest cryptocurrency news relevant to your selected symbols.
            - Saves news articles to the configured database.
            ## 🤖 Model Details
            | Model         | Type       | Data Used                  | Notes |
            |---------------|------------|----------------------------|-------|
            | **ARIMA** | Statistical| Close price only           | Best for linear time series, captures trends and seasonality. |
            | **Linear Regression** | ML         | OHLCV + Indicators         | Simple, fast, and interpretable. Good baseline. |
            | **LSTM** | Deep Learning | OHLCV + Indicators         | Excellent for sequential data, captures long-term dependencies. |
            | **Transformer** | Deep Learning | OHLCV + Indicators         | Powerful for capturing complex relationships and attention mechanisms in time series. |
            | **Ensemble** | Combined   | Predictions from all above | Averages predictions from all trained individual models for robustness. |
            🔒 All models use a configurable lookback period (default 60 steps), scaled inputs (MinMax), and predict a configurable number of future candles.
            ## 📈 How to Interpret Predictions
            - Predicted prices are shown as dashed lines extending from the chart.
            - Use **support/resistance** levels, **trend** analysis, and **indicator signals** to validate the prediction.
            - Combine with **order book** data and **sentiment** for a more confident trading decision.
            ## 🧠 Understanding Buy/Sell Zones
            - Based on thresholds from indicators like RSI and Bollinger Bands.
            - Shown as green/red zones or triangle markers on charts.
            - **Buy Signal**: Often triggered by an oversold RSI, price touching lower Bollinger Band, or a bullish MA crossover.
            - **Sell Signal**: Often triggered by an overbought RSI, price touching upper Bollinger Band, or a bearish MA crossover.
            ## ⚠️ Troubleshooting & Known Issues
            - **No Data?** Ensure the selected symbol/timeframe is valid for the chosen exchange.
            - **Database Connection Failed?** Check your credentials and ensure the database server is accessible. The app will run without persistence if connection fails.
            - **No sentiment?** Ensure a valid NewsAPI key is provided in the sidebar.
            - **Too few candles?** Adjust the "Prediction Horizon" slider or select a longer timeframe in the sidebar.
            - **Mobile UI glitch?** The app includes basic responsiveness, but complex charts may still require desktop for optimal viewing.
            ## 💡 Future Plans
            - Real trading API integration (with secure key management).
            - Advanced alerts/notifications (e.g., Telegram, Email integration).
            - More sophisticated hyperparameter optimization for all models.
            - Comprehensive trading journal exports and CSV reports.
            - Automated strategy backtest optimizer and auto-selector.
            ## 📬 Support
            - Raise a GitHub issue or contact the development team.
            - Suggestions and Pull Requests are always welcome!
            ---
            ⚙️ Built with ❤️ by traders, for traders.
            """
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

        logging.error(f"Critical error in main execution: {e}\n{traceback.format_exc()}")
