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

EXCHANGES = {
    "Binance": ccxt_async.binance,
    "Kraken": ccxt_async.kraken,
    "Bybit": ccxt_async.bybit,
    "MEXC": ccxt_async.mexc,
    "Bitget": ccxt_async.bitget,
    "BingX": ccxt_async.bingx,
}

# --- SENTIMENT ANALYSIS ---
def search_x(query, api_key):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&language=en&pageSize=10&apiKey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return {"posts": [{"text": article.get("description", "") or article.get("title", ""), "url": article.get("url", ""), "source": article.get("source", {}).get("name", "Unknown"), "publishedAt": article.get("publishedAt", "")} for article in articles if article.get("description") or article.get("title")]}
    except Exception as e:
        logging.error(f"Failed to fetch news: {str(e)}")
        return {"posts": []}

def get_sentiment(symbol, api_key):
    try:
        base = symbol.split("/")[0]
        if not api_key:
            return 0.0
        results = search_x(f"{base} crypto", api_key)
        if not results["posts"]:
            return 0.0
        positive_words = ["bullish", "buy", "up", "rise", "gain", "positive", "surge", "rally", "breakout", "strong", "growth"]
        negative_words = ["bearish", "sell", "down", "drop", "loss", "negative", "crash", "decline", "dump", "weak", "fall"]
        score = 0
        count = 0
        for post in results.get("posts", [])[:10]:
            text = post.get("text", "").lower()
            score += sum(1 for word in positive_words if word in text) - sum(1 for word in negative_words if word in text)
            count += 1
        return score / max(count, 1)
    except Exception as e:
        logging.error(f"Failed to fetch sentiment for {symbol}: {str(e)}")
        return 0.0

# --- DATABASE SETUP & UTILITIES ---
def init_db(db_type, host, port, username, password, database):
    try:
        if db_type == "MongoDB":
            client = pymongo.MongoClient(host=host, port=int(port), username=username, password=password)
            db = client[database]
            for col in ["strategy_trades", "news_articles", "preferences"]:
                if col not in db.list_collection_names():
                    db.create_collection(col)
            return db
        elif db_type == "MSSQL":
            conn = pymssql.connect(server=host, port=int(port), user=username, password=password, database=database)
            cursor = conn.cursor(as_dict=True)
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name='preferences')
                CREATE TABLE preferences (
                    user_id VARCHAR(50), exchange VARCHAR(50), symbols TEXT, timeframe VARCHAR(10),
                    indicators TEXT, strategies TEXT, news_api_key VARCHAR(100)
                );
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name='strategy_trades')
                CREATE TABLE strategy_trades (
                    user_id VARCHAR(50), symbol VARCHAR(50), timeframe VARCHAR(10), model_type VARCHAR(50),
                    predictions TEXT, trades TEXT, timestamp VARCHAR(50), start_timestamp VARCHAR(50)
                );
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name='news_articles')
                CREATE TABLE news_articles (
                    user_id VARCHAR(50), symbol VARCHAR(50), title TEXT, description TEXT, url TEXT, source VARCHAR(100), published_at VARCHAR(50), fetched_at VARCHAR(50)
                );
            """)
            conn.commit()
            return conn
        elif db_type == "MySQL":
            conn = mysql.connector.connect(host=host, port=int(port), user=username, password=password, database=database)
            cursor = conn.cursor(dictionary=True)
            for table in ["preferences", "strategy_trades", "news_articles"]:
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} (user_id VARCHAR(50), exchange VARCHAR(50), symbols TEXT, timeframe VARCHAR(10), indicators TEXT, strategies TEXT, news_api_key VARCHAR(100))" if table == "preferences" else
                               f"CREATE TABLE IF NOT EXISTS {table} (user_id VARCHAR(50), symbol VARCHAR(50), timeframe VARCHAR(10), model_type VARCHAR(50), predictions TEXT, trades TEXT, timestamp VARCHAR(50), start_timestamp VARCHAR(50))" if table == "strategy_trades" else
                               f"CREATE TABLE IF NOT EXISTS {table} (user_id VARCHAR(50), symbol VARCHAR(50), title TEXT, description TEXT, url TEXT, source VARCHAR(100), published_at VARCHAR(50), fetched_at VARCHAR(50))")
            conn.commit()
            return conn
        else:
            raise ValueError("Unsupported database type")
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        return None

def save_preferences(db, db_type, user_id, exchange, symbols, timeframe, indicators, strategies, news_api_key):
    try:
        if db_type == "MongoDB":
            db.preferences.update_one(
                {"user_id": user_id},
                {"$set": {"exchange": exchange, "symbols": symbols, "timeframe": timeframe,
                          "indicators": indicators, "strategies": strategies, "news_api_key": news_api_key}},
                upsert=True,
            )
        elif db_type in ["MSSQL", "MySQL"]:
            cursor = db.cursor()
            symbols_str = ",".join(symbols)
            indicators_str = json.dumps(indicators)
            strategies_str = json.dumps(strategies)
            if db_type == "MSSQL":
                cursor.execute(
                    """
                    MERGE INTO preferences AS target
                    USING (SELECT %s, %s, %s, %s, %s, %s, %s) AS source (user_id, exchange, symbols, timeframe, indicators, strategies, news_api_key)
                    ON target.user_id = source.user_id
                    WHEN MATCHED THEN UPDATE SET exchange = source.exchange, symbols = source.symbols, timeframe = source.timeframe,
                                   indicators = source.indicators, strategies = source.strategies, news_api_key = source.news_api_key
                    WHEN NOT MATCHED THEN INSERT (user_id, exchange, symbols, timeframe, indicators, strategies, news_api_key)
                        VALUES (source.user_id, source.exchange, source.symbols, source.timeframe, source.indicators, source.strategies, source.news_api_key);
                """, (user_id, exchange, symbols_str, timeframe, indicators_str, strategies_str, news_api_key))
            else:
                cursor.execute(
                    """
                    INSERT INTO preferences (user_id, exchange, symbols, timeframe, indicators, strategies, news_api_key)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE exchange = %s, symbols = %s, timeframe = %s, indicators = %s, strategies = %s, news_api_key = %s
                """, (user_id, exchange, symbols_str, timeframe, indicators_str, strategies_str, news_api_key,
                        exchange, symbols_str, timeframe, indicators_str, strategies_str, news_api_key))
            db.commit()
    except Exception as e:
        logging.error(f"Failed to save preferences: {str(e)}")

def save_strategy_results(db, db_type, user_id, symbol, timeframe, model_type, predictions, trades, timestamp, start_timestamp):
    try:
        serialized_predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        serialized_trades = []
        for trade in trades:
            trade_copy = trade.copy()
            if isinstance(trade_copy["Date"], pd.Timestamp):
                trade_copy["Date"] = trade_copy["Date"].isoformat()
            serialized_trades.append(trade_copy)
        if db_type == "MongoDB":
            db.strategy_trades.update_one(
                {"user_id": user_id, "symbol": symbol, "timeframe": timeframe, "timestamp": timestamp},
                {"$set": {"model_type": model_type, "predictions": serialized_predictions,
                          "trades": serialized_trades, "timestamp": timestamp, "start_timestamp": start_timestamp}},
                upsert=True,
            )
        elif db_type in ["MSSQL", "MySQL"]:
            cursor = db.cursor()
            trades_str = json.dumps(serialized_trades)
            predictions_str = json.dumps(serialized_predictions)
            if db_type == "MSSQL":
                cursor.execute(
                    """
                    MERGE INTO strategy_trades AS target
                    USING (SELECT %s, %s, %s, %s, %s, %s, %s, %s) AS source (user_id, symbol, timeframe, model_type, predictions, trades, timestamp, start_timestamp)
                    ON target.user_id = source.user_id AND target.symbol = source.symbol AND target.timeframe = source.timeframe AND target.timestamp = source.timestamp
                    WHEN MATCHED THEN UPDATE SET model_type = source.model_type, predictions = source.predictions, trades = source.trades, start_timestamp = source.start_timestamp
                    WHEN NOT MATCHED THEN INSERT (user_id, symbol, timeframe, model_type, predictions, trades, timestamp, start_timestamp)
                        VALUES (source.user_id, source.symbol, source.timeframe, source.model_type, source.predictions, source.trades, source.timestamp, source.start_timestamp);
                """, (user_id, symbol, timeframe, model_type, predictions_str, trades_str, timestamp, start_timestamp))
            else:
                cursor.execute(
                    """
                    INSERT INTO strategy_trades (user_id, symbol, timeframe, model_type, predictions, trades, timestamp, start_timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE model_type = %s, predictions = %s, trades = %s, start_timestamp = %s
                """, (user_id, symbol, timeframe, model_type, predictions_str, trades_str, timestamp, start_timestamp,
                        model_type, predictions_str, trades_str, start_timestamp))
            db.commit()
    except Exception as e:
        logging.error(f"Failed to save strategy results: {str(e)}")

def load_preferences(db, db_type, user_id):
    try:
        if db_type == "MongoDB":
            doc = db.preferences.find_one({"user_id": user_id})
            if doc:
                return (
                    doc.get("exchange", "Binance"), doc.get("symbols", ["BTC/USDT"]),
                    doc.get("timeframe", "1h"), doc.get("indicators", {}),
                    doc.get("strategies", {}), doc.get("news_api_key", ""),
                )
        elif db_type in ["MSSQL", "MySQL"]:
            cursor = db.cursor(as_dict=True) if db_type == "MSSQL" else db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM preferences WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            if result:
                indicators_dict = json.loads(result["indicators"]) if result["indicators"] else {}
                strategies_dict = json.loads(result["strategies"]) if result["strategies"] else {}
                return (
                    result["exchange"],
                    result["symbols"].split(",") if result["symbols"] else [],
                    result["timeframe"],
                    indicators_dict,
                    strategies_dict,
                    result.get("news_api_key", ""),
                )
        return "Binance", ["BTC/USDT"], "1h", {}, {}, ""
    except Exception as e:
        logging.error(f"Failed to load preferences: {str(e)}")
        return "Binance", ["BTC/USDT"], "1h", {}, {}, ""

# --- PORTFOLIO MANAGEMENT ---
def init_portfolio():
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {
            "positions": {},
            "trades": [],
            "balance": st.session_state.get("selected_investment_amount", 100.0),
        }

def update_portfolio(symbol, action, price, shares, strategy, date, entry_price=None):
    portfolio = st.session_state.portfolio
    if action == "Buy":
        portfolio["balance"] -= price * shares
        portfolio["positions"][symbol] = {
            "entry_price": price, "shares": shares, "entry_date": date, "strategy": strategy,
        }
    elif action == "Sell" and symbol in portfolio["positions"]:
        position = portfolio["positions"].pop(symbol)
        profit = (price - position["entry_price"]) * shares
        portfolio["balance"] += price * shares
        portfolio["trades"].append(
            {
                "symbol": symbol, "entry_price": position["entry_price"], "exit_price": price,
                "shares": shares, "profit": profit, "entry_date": position["entry_date"],
                "exit_date": date, "strategy": strategy,
            }
        )

def calculate_portfolio_metrics():
    portfolio = st.session_state.portfolio
    initial_investment = st.session_state.get("selected_investment_amount", 100.0)
    total_profit = sum(trade["profit"] for trade in portfolio["trades"] if trade["profit"] is not None)
    total_return_pct = (total_profit / initial_investment) * 100 if initial_investment > 0 else 0
    returns = [
        trade["profit"] / (trade["entry_price"] * trade["shares"]) * 100
        for trade in portfolio["trades"] if trade["profit"] is not None and trade["entry_price"] * trade["shares"] > 0
    ]
    sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) != 0 else 0)
    equity_curve = [initial_investment]
    current_equity = initial_investment
    for trade in portfolio["trades"]:
        if trade["profit"] is not None:
            current_equity += trade["profit"]
        equity_curve.append(current_equity)
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 1 else 0
    return total_return_pct, sharpe_ratio, max_drawdown, equity_curve

# --- REAL-TIME & CACHED DATA FETCHING ---
@st.cache_data(ttl=600)
def get_ohlcv(exchange_name, symbol, timeframe, limit=200):
    async def fetch_ohlcv_async(exchange_name, symbol, timeframe, limit):
        exchange_constructor = EXCHANGES.get(exchange_name)
        if not exchange_constructor:
            return pd.DataFrame()
        exchange = exchange_constructor()
        try:
            await exchange.load_markets()
            if symbol not in exchange.markets or not exchange.markets[symbol].get("active", False):
                return pd.DataFrame()
            data = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = symbol
            return df
        except Exception as e:
            logging.error(f"Failed to fetch OHLCV data for {symbol} on {exchange_name}: {str(e)}")
            return pd.DataFrame()
        finally:
            try:
                await exchange.close()
            except Exception:
                pass
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(fetch_ohlcv_async(exchange_name, symbol, timeframe, limit))
        return result
    finally:
        loop.close()

# --- EXTERNAL DATA ---
async def fetch_external_data(exchange_name, symbol, news_api_key):
    funding_rate = "N/A"
    order_book_imbalance = None
    sentiment_score = 0.0
    exchange_constructor = EXCHANGES.get(exchange_name)
    if not exchange_constructor:
        return funding_rate, order_book_imbalance, sentiment_score
    exchange = exchange_constructor()
    try:
        await exchange.load_markets()
        if exchange.has["fetchFundingRate"] and symbol in exchange.markets and exchange.markets[symbol].get("type") in ["future", "swap"]:
            try:
                funding = await exchange.fetch_funding_rate(symbol)
                funding_rate = f"{funding.get('fundingRate', 0.0) * 100:.4f}%"
            except Exception as e:
                logging.warning(f"Failed to fetch funding rate for {symbol}: {e}")
        if exchange.has["fetchOrderBook"]:
            try:
                order_book = await exchange.fetch_order_book(symbol, limit=20)
                bid_volumes = sum(b[1] for b in order_book["bids"])
                ask_volumes = sum(a[1] for a in order_book["asks"])
                if (bid_volumes + ask_volumes) > 0:
                    order_book_imbalance = bid_volumes / (bid_volumes + ask_volumes)
                else:
                    order_book_imbalance = 0.5
            except Exception as e:
                logging.warning(f"Failed to fetch order book for {symbol}: {e}")
        sentiment_score = get_sentiment(symbol, news_api_key)
    except Exception as e:
        logging.warning(f"Failed to fetch external data for {symbol}: {e}")
    finally:
        try:
            await exchange.close()
        except Exception:
            pass
    return funding_rate, order_book_imbalance, sentiment_score

# --- ORDER BOOK VISUALIZATION ---
async def fetch_order_book(exchange_name, symbol, depth):
    exchange_constructor = EXCHANGES.get(exchange_name)
    if not exchange_constructor:
        return pd.DataFrame(), pd.DataFrame(), None, None
    exchange = exchange_constructor()
    try:
        await exchange.load_markets()
        order_book = await exchange.fetch_order_book(symbol, limit=depth)
        bids = pd.DataFrame(order_book["bids"], columns=["price", "volume"])[:depth]
        asks = pd.DataFrame(order_book["asks"], columns=["price", "volume"])[:depth]
        spread = asks["price"].iloc[0] - bids["price"].iloc[0] if not bids.empty and not asks.empty else None
        imbalance = (
            bids["volume"].sum() / (bids["volume"].sum() + asks["volume"].sum())
            if bids["volume"].sum() + asks["volume"].sum() > 0
            else 0.5
        )
        return bids, asks, spread, imbalance
    except Exception as e:
        logging.error(f"Failed to fetch order book for {symbol}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), None, None
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

def plot_order_book(bids, asks, symbol, theme, custom_colors):
    try:
        colors = custom_colors or {"bg": "#FFFFFF", "fg": "#000000", "grid": "rgba(128, 128, 128, 0.2)"}
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bids["price"], y=bids["volume"].cumsum(), name="Bids", fill="tozeroy", line=dict(color="green"), mode='lines'))
        fig.add_trace(go.Scatter(x=asks["price"], y=asks["volume"].cumsum(), name="Asks", fill="tozeroy", line=dict(color="red"), mode='lines'))
        height = 600 if st.session_state.get("is_mobile", False) else 800
        fig.update_layout(
            title=f"{symbol} Order Book",
            xaxis_title="Price (USDT)",
            yaxis_title="Cumulative Volume",
            template="plotly_dark" if theme == "Dark" else "plotly_white",
            paper_bgcolor=colors["bg"],
            plot_bgcolor=colors["bg"],
            font=dict(color=colors["fg"]),
            xaxis=dict(gridcolor=colors["grid"]),
            yaxis=dict(gridcolor=colors["grid"]),
            height=height,
            hovermode="x unified",
        )
        return fig
    except Exception as e:
        logging.error(f"Error plotting order book for {symbol}: {str(e)}\n{traceback.format_exc()}")
        return go.Figure()

# --- INDICATOR PROCESSOR ---
def add_indicators(df, preds=None, extend_vwap=False, extend_bb=False, indicators=None, params=None):
    if df.empty:
        return df
    df = df.copy()
    if indicators is None:
        indicators = {}
    if params is None:
        params = {}
    if "funding_rate" in df.columns:
        df["Funding_Rate_Feature"] = pd.to_numeric(df["funding_rate"].str.replace('%', ''), errors='coerce').fillna(0) / 0.0001
    if "order_book_imbalance" in df.columns:
        df["Order_Book_Imbalance_Feature"] = df["order_book_imbalance"].fillna(0.5)
    if "sentiment_score" in df.columns:
        df["Sentiment_Score_Feature"] = df["sentiment_score"].fillna(0)
    if "close" in df.columns and "volume" in df.columns and "high" in df.columns and "low" in df.columns:
        if indicators.get("MA20/50", False):
            df["MA20"] = df["close"].rolling(window=params.get("ma_short", 20)).mean()
            df["MA50"] = df["close"].rolling(window=params.get("ma_long", 50)).mean()
        if indicators.get("Bollinger Bands", False):
            if len(df) >= params.get("bb_window", 20):
                rolling_mean = df["close"].rolling(window=params.get("bb_window", 20)).mean()
                rolling_std = df["close"].rolling(window=params.get("bb_window", 20)).std()
                df["BB_High"] = rolling_mean + (rolling_std * params.get("bb_std", 2.0))
                df["BB_Low"] = rolling_mean - (rolling_std * params.get("bb_std", 2.0))
            else:
                df["BB_High"] = np.nan
                df["BB_Low"] = np.nan
        if indicators.get("RSI Zones", False):
            if len(df) >= params.get("rsi_period", 14):
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                df["Buy_Zone"] = (df["RSI"] < params.get("alert_rsi_oversold", 30.0)).astype(int)
                df["Sell_Zone"] = (df["RSI"] > params.get("alert_rsi_overbought", 70.0)).astype(int)
            else:
                df["RSI"] = np.nan
                df["Buy_Zone"] = 0
                df["Sell_Zone"] = 0
        if indicators.get("MACD", False):
            if len(df) >= max(params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9)):
                ema_fast = df["close"].ewm(span=params.get("macd_fast", 12), adjust=False).mean()
                ema_slow = df["close"].ewm(span=params.get("macd_slow", 26), adjust=False).mean()
                df["MACD"] = ema_fast - ema_slow
                df["MACD_Signal"] = df["MACD"].ewm(span=params.get("macd_signal", 9), adjust=False).mean()
                df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
            else:
                df["MACD"] = np.nan
                df["MACD_Signal"] = np.nan
                df["MACD_Hist"] = np.nan
        if indicators.get("VWAP", False):
            if not df["volume"].isnull().all() and df["volume"].sum() > 0:
                df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            else:
                df["VWAP"] = np.nan
        if indicators.get("Stochastic", False):
            if len(df) >= 14:
                lowest_low = df["low"].rolling(window=14).min()
                highest_high = df["high"].rolling(window=14).max()
                df["Stoch_K"] = ((df["close"] - lowest_low) / (highest_high - lowest_low)) * 100
                df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
            else:
                df["Stoch_K"] = np.nan
                df["Stoch_D"] = np.nan
        if indicators.get("ATR", False):
            if len(df) >= 14:
                high_minus_low = df["high"] - df["low"]
                high_minus_prev_close = abs(df["high"] - df["close"].shift(1))
                low_minus_prev_close = abs(df["low"] - df["close"].shift(1))
                true_range = pd.DataFrame({'hml': high_minus_low, 'hmc': high_minus_prev_close, 'lmc': low_minus_prev_close}).max(axis=1)
                df["ATR"] = true_range.rolling(window=14).mean()
            else:
                df["ATR"] = np.nan
        if indicators.get("Momentum", False):
            df["Momentum"] = df["close"].pct_change(periods=params.get("mom_period", 5))
        df["Volume_Spike"] = (df["volume"] > df["volume"].rolling(20).mean() * 1.5).astype(int)
    if not df.empty and len(df) >= 20:
        high_20 = df["high"].rolling(20).max().iloc[-1]
        low_20 = df["low"].rolling(20).min().iloc[-1]
        diff = high_20 - low_20
        if diff > 0:
            df["Fib_23.6"] = high_20 - diff * 0.236
            df["Fib_38.2"] = high_20 - diff * 0.382
            df["Fib_50"] = high_20 - diff * 0.5
            df["Fib_61.8"] = high_20 - diff * 0.618
        else:
            df["Fib_23.6"] = df["Fib_38.2"] = df["Fib_50"] = df["Fib_61.8"] = np.nan
    else:
        df["Fib_23.6"] = df["Fib_38.2"] = df["Fib_50"] = df["Fib_61.8"] = np.nan
    if preds is not None and len(preds) > 0:
        freq_map = {"1m": "T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "H", "4h": "4H", "1d": "D"}
        freq = freq_map.get(st.session_state.get("selected_timeframe", "1h"), "T")
        start_timestamp = df["timestamp"].iloc[-1] if not df.empty else pd.Timestamp.now()
        future_x = pd.date_range(start_timestamp, periods=len(preds) + 1, freq=freq)[1:]
        mean_vol = df["volume"].mean() if not df.empty else 1000
        vol_trend = df["volume"].pct_change().mean() if extend_vwap or extend_bb and not df.empty else 0
        vol_preds = [mean_vol * (1 + vol_trend * i) for i in range(len(preds))]
        open_prices_for_preds = [df["close"].iloc[-1]] + list(preds[:-1]) if not df.empty else [preds[0]] + list(preds[:-1])
        pred_df = pd.DataFrame(
            {
                "timestamp": future_x,
                "close": preds,
                "open": open_prices_for_preds,
                "high": [p + p * 0.01 for p in preds],
                "low": [p - p * 0.01 for p in preds],
                "volume": vol_preds,
                "symbol": df["symbol"].iloc[-1] if not df.empty else "N/A"
            }
        )
        df = pd.concat([df, pred_df], ignore_index=True)
        if indicators.get("MA20/50", False):
            df["MA20"] = df["close"].rolling(window=params.get("ma_short", 20)).mean()
            df["MA50"] = df["close"].rolling(window=params.get("ma_long", 50)).mean()
        if indicators.get("RSI Zones", False):
            if len(df) >= params.get("rsi_period", 14):
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
            else:
                df["RSI"] = np.nan
        if indicators.get("MACD", False):
            if len(df) >= max(params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9)):
                ema_fast = df["close"].ewm(span=params.get("macd_fast", 12), adjust=False).mean()
                ema_slow = df["close"].ewm(span=params.get("macd_slow", 26), adjust=False).mean()
                df["MACD"] = ema_fast - ema_slow
                df["MACD_Signal"] = df["MACD"].ewm(span=params.get("macd_signal", 9), adjust=False).mean()
                df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
            else:
                df["MACD"] = np.nan
                df["MACD_Signal"] = np.nan
                df["MACD_Hist"] = np.nan
        if indicators.get("Stochastic", False):
            if len(df) >= 14:
                lowest_low = df["low"].rolling(window=14).min()
                highest_high = df["high"].rolling(window=14).max()
                df["Stoch_K"] = ((df["close"] - lowest_low) / (highest_high - lowest_low)) * 100
                df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
            else:
                df["Stoch_K"] = np.nan
                df["Stoch_D"] = np.nan
        if indicators.get("ATR", False):
            if len(df) >= 14:
                high_minus_low = df["high"] - df["low"]
                high_minus_prev_close = abs(df["high"] - df["close"].shift(1))
                low_minus_prev_close = abs(df["low"] - df["close"].shift(1))
                true_range = pd.DataFrame({'hml': high_minus_low, 'hmc': high_minus_prev_close, 'lmc': low_minus_prev_close}).max(axis=1)
                df["ATR"] = true_range.rolling(window=14).mean()
            else:
                df["ATR"] = np.nan
        if extend_vwap and indicators.get("VWAP", False):
            if not df["volume"].isnull().all() and df["volume"].sum() > 0:
                df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            else:
                df["VWAP"] = np.nan
        if extend_bb and indicators.get("Bollinger Bands", False):
            if len(df) >= params.get("bb_window", 20):
                rolling_mean = df["close"].rolling(window=params.get("bb_window", 20)).mean()
                rolling_std = df["close"].rolling(window=params.get("bb_window", 20)).std()
                df["BB_High"] = rolling_mean + (rolling_std * params.get("bb_std", 2.0))
                df["BB_Low"] = rolling_mean - (rolling_std * params.get("bb_std", 2.0))
            else:
                df["BB_High"] = np.nan
                df["BB_Low"] = np.nan
    return df

def detect_trend(df):
    if df.empty or "MA20" not in df.columns or "MA50" not in df.columns:
        return "Unknown"
    if pd.isna(df["MA20"].iloc[-1]) or pd.isna(df["MA50"].iloc[-1]):
        return "Unknown"
    if df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
        return "Uptrend"
    elif df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
        return "Downtrend"
    return "Sideways"

def support_resistance(df):
    if df.empty:
        return None, None
    if len(df) >= 20:
        support = df["low"].rolling(window=20).min().iloc[-1]
        resistance = df["high"].rolling(window=20).max().iloc[-1]
    else:
        support = df["low"].min()
        resistance = df["high"].max()
    return round(support, 2) if pd.notnull(support) else None, round(resistance, 2) if pd.notnull(resistance) else None

# --- TRADING STRATEGIES ---
def moving_average_crossover(df, investment, params=None):
    if params is None: params = {}
    if df.empty or "MA20" not in df.columns or "MA50" not in df.columns: return []
    df = df.copy()
    df["Signal"] = 0
    df.loc[df["MA20"] > df["MA50"], "Signal"] = 1
    df.loc[df["MA20"] < df["MA50"], "Signal"] = -1
    trades = []
    position = 0
    entry_price = 0
    shares = 0
    stop_loss = 0
    take_profit = 0
    for i in range(1, len(df)):
        if df["Signal"].iloc[i-1] == 1 and position == 0:
            entry_price = df["close"].iloc[i]
            if entry_price == 0: continue
            shares = investment / entry_price
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.10
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Buy", "Price": entry_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": stop_loss, "Take_Profit": take_profit, "Profit": None, "Profit_Pct": None, "Strategy": "Moving Average Crossover"})
            update_portfolio(df["symbol"].iloc[0], "Buy", entry_price, shares, "Moving Average Crossover", df["timestamp"].iloc[i])
            position = 1
        elif (position == 1 and (df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit or df["Signal"].iloc[i] == -1)):
            exit_price = df["close"].iloc[i]
            profit = (exit_price - entry_price) * shares
            profit_pct = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Sell", "Price": exit_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": None, "Take_Profit": None, "Profit": profit, "Profit_Pct": profit_pct, "Strategy": "Moving Average Crossover"})
            update_portfolio(df["symbol"].iloc[0], "Sell", exit_price, shares, "Moving Average Crossover", df["timestamp"].iloc[i])
            position = 0
    return trades

def mean_reversion(df, investment, params=None):
    if params is None: params = {}
    if df.empty: return []
    df = df.copy()
    if len(df) < params.get("mr_window", 20): return []
    df["Mean"] = df["close"].rolling(window=params.get("mr_window", 20)).mean()
    df["Std"] = df["close"].rolling(window=params.get("mr_window", 20)).std()
    df["Signal"] = 0
    df.loc[df["close"] < (df["Mean"] - params.get("mr_std", 1.5) * df["Std"]), "Signal"] = 1
    df.loc[df["close"] > (df["Mean"] + params.get("mr_std", 1.5) * df["Std"]), "Signal"] = -1
    trades = []
    position = 0
    entry_price = 0
    shares = 0
    stop_loss = 0
    take_profit = 0
    for i in range(1, len(df)):
        if df["Signal"].iloc[i-1] == 1 and position == 0:
            entry_price = df["close"].iloc[i]
            if entry_price == 0: continue
            shares = investment / entry_price
            stop_loss = entry_price * 0.93
            take_profit = entry_price * 1.12
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Buy", "Price": entry_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": stop_loss, "Take_Profit": take_profit, "Profit": None, "Profit_Pct": None, "Strategy": "Mean Reversion"})
            update_portfolio(df["symbol"].iloc[0], "Buy", entry_price, shares, "Mean Reversion", df["timestamp"].iloc[i])
            position = 1
        elif (position == 1 and (df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit or df["Signal"].iloc[i] == -1)):
            exit_price = df["close"].iloc[i]
            profit = (exit_price - entry_price) * shares
            profit_pct = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Sell", "Price": exit_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": None, "Take_Profit": None, "Profit": profit, "Profit_Pct": profit_pct, "Strategy": "Mean Reversion"})
            update_portfolio(df["symbol"].iloc[0], "Sell", exit_price, shares, "Mean Reversion", df["timestamp"].iloc[i])
            position = 0
    return trades

def momentum_trading(df, investment, params=None):
    if params is None: params = {}
    if df.empty: return []
    df = df.copy()
    if len(df) < params.get("mom_period", 5): return []
    df["Momentum"] = df["close"].pct_change(periods=params.get("mom_period", 5))
    df["Signal"] = 0
    df.loc[df["Momentum"] > params.get("mom_threshold", 0.05), "Signal"] = 1
    df.loc[df["Momentum"] < -params.get("mom_threshold", 0.05), "Signal"] = -1
    trades = []
    position = 0
    entry_price = 0
    shares = 0
    stop_loss = 0
    take_profit = 0
    for i in range(1, len(df)):
        if df["Signal"].iloc[i-1] == 1 and position == 0:
            entry_price = df["close"].iloc[i]
            if entry_price == 0: continue
            shares = investment / entry_price
            stop_loss = entry_price * 0.94
            take_profit = entry_price * 1.15
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Buy", "Price": entry_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": stop_loss, "Take_Profit": take_profit, "Profit": None, "Profit_Pct": None, "Strategy": "Momentum"})
            update_portfolio(df["symbol"].iloc[0], "Buy", entry_price, shares, "Momentum", df["timestamp"].iloc[i])
            position = 1
        elif (position == 1 and (df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit or df["Signal"].iloc[i] == -1)):
            exit_price = df["close"].iloc[i]
            profit = (exit_price - entry_price) * shares
            profit_pct = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Sell", "Price": exit_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": None, "Take_Profit": None, "Profit": profit, "Profit_Pct": profit_pct, "Strategy": "Momentum"})
            update_portfolio(df["symbol"].iloc[0], "Sell", exit_price, shares, "Momentum", df["timestamp"].iloc[i])
            position = 0
    return trades

def bollinger_bands_strategy(df, investment, params=None):
    if params is None: params = {}
    if df.empty or "BB_Low" not in df.columns or "BB_High" not in df.columns: return []
    df = df.copy()
    df["Signal"] = 0
    df.loc[df["close"] < df["BB_Low"], "Signal"] = 1
    df.loc[df["close"] > df["BB_High"], "Signal"] = -1
    trades = []
    position = 0
    entry_price = 0
    shares = 0
    stop_loss = 0
    take_profit = 0
    for i in range(1, len(df)):
        if df["Signal"].iloc[i-1] == 1 and position == 0:
            entry_price = df["close"].iloc[i]
            if entry_price == 0: continue
            shares = investment / entry_price
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.10
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Buy", "Price": entry_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": stop_loss, "Take_Profit": take_profit, "Profit": None, "Profit_Pct": None, "Strategy": "Bollinger Bands"})
            update_portfolio(df["symbol"].iloc[0], "Buy", entry_price, shares, "Bollinger Bands", df["timestamp"].iloc[i])
            position = 1
        elif (position == 1 and (df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit or df["Signal"].iloc[i] == -1)):
            exit_price = df["close"].iloc[i]
            profit = (exit_price - entry_price) * shares
            profit_pct = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Sell", "Price": exit_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": None, "Take_Profit": None, "Profit": profit, "Profit_Pct": profit_pct, "Strategy": "Bollinger Bands"})
            update_portfolio(df["symbol"].iloc[0], "Sell", exit_price, shares, "Bollinger Bands", df["timestamp"].iloc[i])
            position = 0
    return trades

def rsi_divergence_strategy(df, investment, params=None):
    if params is None: params = {}
    if df.empty or len(df) < params.get("rsi_period", 14) + 10 or "RSI" not in df.columns: return []
    df = df.copy()
    df["Price_Diff"] = df["close"].diff()
    df["RSI_Diff"] = df["RSI"].diff()
    df["Signal"] = 0
    for i in range(params.get("rsi_period", 14) + 1, len(df)):
        if df["Price_Diff"].iloc[i] > 0 and df["RSI_Diff"].iloc[i] < 0 and df["RSI"].iloc[i] < params.get("alert_rsi_oversold", 30.0):
            df.loc[i, "Signal"] = 1
        elif df["Price_Diff"].iloc[i] < 0 and df["RSI_Diff"].iloc[i] > 0 and df["RSI"].iloc[i] > params.get("alert_rsi_overbought", 70.0):
            df.loc[i, "Signal"] = -1
    trades = []
    position = 0
    entry_price = 0
    shares = 0
    stop_loss = 0
    take_profit = 0
    for i in range(1, len(df)):
        if df["Signal"].iloc[i-1] == 1 and position == 0:
            entry_price = df["close"].iloc[i]
            if entry_price == 0: continue
            shares = investment / entry_price
            stop_loss = entry_price * 0.90
            take_profit = entry_price * 1.20
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Buy", "Price": entry_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": stop_loss, "Take_Profit": take_profit, "Profit": None, "Profit_Pct": None, "Strategy": "RSI Divergence"})
            update_portfolio(df["symbol"].iloc[0], "Buy", entry_price, shares, "RSI Divergence", df["timestamp"].iloc[i])
            position = 1
        elif (position == 1 and (df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit or df["Signal"].iloc[i] == -1)):
            exit_price = df["close"].iloc[i]
            profit = (exit_price - entry_price) * shares
            profit_pct = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Sell", "Price": exit_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": None, "Take_Profit": None, "Profit": profit, "Profit_Pct": profit_pct, "Strategy": "RSI Divergence"})
            update_portfolio(df["symbol"].iloc[0], "Sell", exit_price, shares, "RSI Divergence", df["timestamp"].iloc[i])
            position = 0
    return trades

def combined_indicators_strategy(df, investment, params=None, indicators=None):
    if params is None: params = {}
    if indicators is None: indicators = {}
    if df.empty: return []
    df = df.copy()
    df["Signal"] = 0
    signals = []
    enabled_indicators = sum(1 for v in indicators.values() if v)
    if enabled_indicators == 0: return []
    if indicators.get("MA20/50", False) and "MA20" in df.columns and "MA50" in df.columns:
        df.loc[df["MA20"] > df["MA50"], "MA_Signal"] = 1
        df.loc[df["MA20"] < df["MA50"], "MA_Signal"] = -1
        signals.append("MA_Signal")
    if indicators.get("Bollinger Bands", False) and "BB_Low" in df.columns and "BB_High" in df.columns:
        df.loc[df["close"] < df["BB_Low"], "BB_Signal"] = 1
        df.loc[df["close"] > df["BB_High"], "BB_Signal"] = -1
        signals.append("BB_Signal")
    if indicators.get("RSI Zones", False) and "RSI" in df.columns:
        df.loc[df["RSI"] < params.get("alert_rsi_oversold", 30.0), "RSI_Signal"] = 1
        df.loc[df["RSI"] > params.get("alert_rsi_overbought", 70.0), "RSI_Signal"] = -1
        signals.append("RSI_Signal")
    if indicators.get("MACD", False) and "MACD" in df.columns and "MACD_Signal" in df.columns:
        df.loc[(df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)), "MACD_Signal"] = 1
        df.loc[(df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)), "MACD_Signal"] = -1
        signals.append("MACD_Signal")
    if indicators.get("Stochastic", False) and "Stoch_K" in df.columns and "Stoch_D" in df.columns:
        df.loc[(df["Stoch_K"] < 20) & (df["Stoch_K"] > df["Stoch_D"]), "Stoch_Signal"] = 1
        df.loc[(df["Stoch_K"] > 80) & (df["Stoch_K"] < df["Stoch_D"]), "Stoch_Signal"] = -1
        signals.append("Stoch_Signal")
    if indicators.get("ATR", False) and "ATR" in df.columns:
        atr_threshold = df["ATR"].mean() * 1.5 if not df["ATR"].isnull().all() else 0
        df.loc[df["close"].diff() > atr_threshold, "ATR_Signal"] = 1
        df.loc[df["close"].diff() < -atr_threshold, "ATR_Signal"] = -1
        signals.append("ATR_Signal")
    if signals:
        df["Consensus_Signal"] = df[signals].mean(axis=1)
        df.loc[df["Consensus_Signal"] >= 0.6, "Signal"] = 1
        df.loc[df["Consensus_Signal"] <= -0.6, "Signal"] = -1
    trades = []
    position = 0
    entry_price = 0
    shares = 0
    stop_loss = 0
    take_profit = 0
    for i in range(1, len(df)):
        if df["Signal"].iloc[i-1] == 1 and position == 0:
            entry_price = df["close"].iloc[i]
            if entry_price == 0: continue
            shares = investment / entry_price
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.10
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Buy", "Price": entry_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": stop_loss, "Take_Profit": take_profit, "Profit": None, "Profit_Pct": None, "Strategy": "Combined Indicators"})
            update_portfolio(df["symbol"].iloc[0], "Buy", entry_price, shares, "Combined Indicators", df["timestamp"].iloc[i])
            position = 1
        elif (position == 1 and (df["close"].iloc[i] <= stop_loss or df["close"].iloc[i] >= take_profit or df["Signal"].iloc[i] == -1)):
            exit_price = df["close"].iloc[i]
            profit = (exit_price - entry_price) * shares
            profit_pct = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0
            trades.append({"Date": df["timestamp"].iloc[i], "Action": "Sell", "Price": exit_price, "Entry_Price": entry_price, "Shares": shares, "Stop_Loss": None, "Take_Profit": None, "Profit": profit, "Profit_Pct": profit_pct, "Strategy": "Combined Indicators"})
            update_portfolio(df["symbol"].iloc[0], "Sell", exit_price, shares, "Combined Indicators", df["timestamp"].iloc[i])
            position = 0
    return trades

# --- MACHINE LEARNING MODELS ---
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = Sequential([
            Dense(embed_dim * 4, activation="relu"),
            Dense(embed_dim),
        ])
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    embed_dim = 32
    x = Dense(embed_dim, activation="relu")(inputs)
    num_heads = 2
    x = TransformerBlock(embed_dim, num_heads)(x, training=False)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

class ARIMAModel:
    def __init__(self, order=(5,1,0)):
        self.order = order
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
    def fit(self, data):
        if data.ndim > 1:
            data = data.flatten()
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        if len(scaled_data) < max(self.order):
            raise ValueError(f"Not enough data points ({len(scaled_data)}) for ARIMA order {self.order}")
        self.model = ARIMA(scaled_data, order=self.order)
        self.model_fit = self.model.fit()
    def predict(self, n_periods):
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        start_idx = len(self.model_fit.fittedvalues)
        end_idx = start_idx + n_periods - 1
        forecast = self.model_fit.predict(start=start_idx, end=end_idx)
        return self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()
        elif y.ndim == 1:
            y = y.reshape(-1, 1)
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        self.model.fit(X_scaled, y_scaled)
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_scaled = self.scaler_X.transform(X)
        predictions_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(predictions_scaled.reshape(-1,1)).flatten()

# --- MODEL TRAINING AND PREDICTION ---
def train_and_predict_model(model_type, df, n_future=5, epochs=10, batch_size=32, tuning_epochs=5):
    try:
        if df.empty:
            return np.array([]), None, None, None, f"Empty DataFrame for {model_type}."
        features = ["close", "open", "high", "low", "volume"]
        if "funding_rate" in df.columns:
            features.append("Funding_Rate_Feature")
        if "order_book_imbalance" in df.columns:
            features.append("Order_Book_Imbalance_Feature")
        if "sentiment_score" in df.columns:
            features.append("Sentiment_Score_Feature")
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return np.array([]), None, None, None, "No valid features available."
        df.dropna(subset=available_features, inplace=True)
        df.reset_index(drop=True, inplace=True)
        X = df[available_features].values
        y = df["close"].values
        lookback = 60
        if len(X) < lookback + 1:
            return np.array([]), None, None, None, f"Not enough data ({len(X)} rows) for lookback {lookback}."
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        X_train, y_train = [], []
        for i in range(lookback, len(X_scaled)):
            X_train.append(X_scaled[i - lookback:i])
            y_train.append(y_scaled[i])
        if not X_train:
            return np.array([]), None, None, None, "Not enough data to create training sequences."
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if len(X_train) == 0:
            return np.array([]), None, None, None, "No training samples after preprocessing."
        if model_type == "Ensemble":
            models = ["LSTM", "Transformer", "ARIMA", "Linear Regression"]
            predictions_all = []
            for model in models:
                cache_key = f"predictions_{df['symbol'].iloc[0]}_{model}_{st.session_state.selected_timeframe}_{n_future}"
                if cache_key in st.session_state:
                    predictions_all.append(st.session_state[cache_key]["predictions"])
                else:
                    sub_predictions, sub_model, sub_rmse, sub_mae, sub_error = train_and_predict_model(
                        model, df, n_future, epochs, batch_size, tuning_epochs
                    )
                    if sub_error:
                        continue
                    st.session_state[cache_key] = {
                        "predictions": sub_predictions,
                        "trained_model": sub_model,
                        "rmse": sub_rmse,
                        "mae": sub_mae
                    }
                    predictions_all.append(sub_predictions)
            if not predictions_all:
                return np.array([]), None, None, None, "No individual models trained."
            min_len = min(len(p) for p in predictions_all)
            predictions_all_trimmed = [p[:min_len] for p in predictions_all]
            predictions_scaled = np.mean(predictions_all_trimmed, axis=0)
            rmse = np.mean([st.session_state[f"predictions_{df['symbol'].iloc[0]}_{m}_{st.session_state.selected_timeframe}_{n_future}"]["rmse"] for m in models if f"predictions_{df['symbol'].iloc[0]}_{m}_{st.session_state.selected_timeframe}_{n_future}" in st.session_state])
            mae = np.mean([st.session_state[f"predictions_{df['symbol'].iloc[0]}_{m}_{st.session_state.selected_timeframe}_{n_future}"]["mae"] for m in models if f"predictions_{df['symbol'].iloc[0]}_{m}_{st.session_state.selected_timeframe}_{n_future}" in st.session_state])
            return predictions_scaled, None, rmse, mae, None
        elif model_type == "LSTM":
            model = build_lstm_model(input_shape=(lookback, X_train.shape[2]))
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            X_pred = X_scaled[-lookback:].reshape(1, lookback, X_train.shape[2])
            predictions_scaled = []
            for _ in range(n_future):
                pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
                predictions_scaled.append(pred_scaled)
                X_pred = np.roll(X_pred, -1, axis=1)
                X_pred[0, -1, 0:4] = pred_scaled
            predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
            y_pred = model.predict(X_train, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            mae = mean_absolute_error(y_train, y_pred)
            return predictions, model, rmse, mae, None
        elif model_type == "Transformer":
            model = build_transformer_model(input_shape=(lookback, X_train.shape[2]))
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            X_pred = X_scaled[-lookback:].reshape(1, lookback, X_train.shape[2])
            predictions_scaled = []
            for _ in range(n_future):
                pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
                predictions_scaled.append(pred_scaled)
                X_pred = np.roll(X_pred, -1, axis=1)
                X_pred[0, -1, 0:4] = pred_scaled
            predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
            y_pred = model.predict(X_train, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            mae = mean_absolute_error(y_train, y_pred)
            return predictions, model, rmse, mae, None
        elif model_type == "ARIMA":
            model = ARIMAModel(order=(5, 1, 0))
            model.fit(y)
            predictions = model.predict(n_future)
            y_pred_scaled = model.model_fit.fittedvalues
            y_train_scaled = model.scaler.transform(y_train.reshape(-1, 1)).flatten()
            comparison_len = min(len(y_pred_scaled), len(y_train_scaled))
            rmse = np.sqrt(mean_squared_error(y_train_scaled[-comparison_len:], y_pred_scaled[-comparison_len:]))
            mae = mean_absolute_error(y_train_scaled[-comparison_len:], y_pred_scaled[-comparison_len:])
            return predictions, model, rmse, mae, None
        elif model_type == "Linear Regression":
            model = LinearRegressionModel()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            model.fit(X_train_flat, y_train)
            X_pred_sequence = X_scaled[-lookback:].reshape(1, lookback, X_train.shape[2])
            predictions_scaled = []
            for _ in range(n_future):
                current_input_flat = X_pred_sequence.reshape(1, -1)
                pred_scaled = model.predict(current_input_flat)[0]
                predictions_scaled.append(pred_scaled)
                X_pred_sequence = np.roll(X_pred_sequence, -1, axis=1)
                X_pred_sequence[0, -1, 0:4] = pred_scaled
            predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
            y_pred = model.predict(X_train_flat)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            mae = mean_absolute_error(y_train, y_pred)
            return predictions, model, rmse, mae, None
        else:
            return np.array([]), None, None, None, f"Unknown model type: {model_type}"
    except Exception as e:
        logging.error(f"Error in train_and_predict_model for {model_type}: {str(e)}\n{traceback.format_exc()}")
        return np.array([]), None, None, None, f"Error training {model_type}: {str(e)}"

# --- PLOTTING FUNCTIONS ---
def plot_candlestick(df, symbol, indicators_to_plot, preds=None, confidence_interval=None, theme="Dark", custom_colors=None):
    colors = custom_colors or {"bg": "#1E1E1E", "fg": "#E0E0E0", "grid": "rgba(128, 128, 128, 0.2)"}
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df["timestamp"],
                                 open=df["open"],
                                 high=df["high"],
                                 low=df["low"],
                                 close=df["close"],
                                 name="Candlesticks",
                                 increasing_line_color='green', decreasing_line_color='red'), row=1, col=1)
    if preds is not None and len(preds) > 0:
        freq_map = {"1m": "T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "H", "4h": "4H", "1d": "D"}
        freq = freq_map.get(st.session_state.get("selected_timeframe", "1h"), "T")
        start_timestamp = df["timestamp"].iloc[-1] if not df.empty else pd.Timestamp.now()
        pred_timestamps = pd.date_range(start_timestamp, periods=len(preds) + 1, freq=freq)[1:]
        fig.add_trace(go.Scatter(x=pred_timestamps, y=preds, mode='lines', name='Predictions',
                                 line=dict(color='blue', dash='dash'), showlegend=True), row=1, col=1)
        if confidence_interval is not None and len(confidence_interval) == len(preds):
            fig.add_trace(go.Scatter(x=pred_timestamps, y=preds + confidence_interval, mode='lines',
                                     line=dict(width=0), name='Upper CI', showlegend=False,
                                     hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=pred_timestamps, y=preds - confidence_interval, mode='lines',
                                     fill='tonexty', fillcolor='rgba(0,100,80,0.2)', name='Lower CI',
                                     showlegend=False, hoverinfo='skip'), row=1, col=1)
    if indicators_to_plot.get("MA20/50", False) and "MA20" in df.columns and "MA50" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA20"], mode='lines', name='MA20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA50"], mode='lines', name='MA50', line=dict(color='purple')), row=1, col=1)
    if indicators_to_plot.get("Bollinger Bands", False) and "BB_High" in df.columns and "BB_Low" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BB_High"], mode='lines', name='BB Upper', line=dict(color='cyan', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BB_Low"], mode='lines', name='BB Lower', line=dict(color='cyan', dash='dot')), row=1, col=1)
    fib_levels = ["Fib_23.6", "Fib_38.2", "Fib_50", "Fib_61.8"]
    for level_col in fib_levels:
        if level_col in df.columns and pd.notna(df[level_col].iloc[-1]):
            fig.add_hline(y=df[level_col].iloc[-1], line_dash="dash", line_color="grey",
                          annotation_text=f"Fib {level_col.split('_')[1]}%", annotation_position="top right", row=1, col=1)
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], name='Volume', marker_color='grey'), row=2, col=1)
    if indicators_to_plot.get("ATR", False) and "ATR" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ATR"], mode='lines', name='ATR', line=dict(color='brown')), row=2, col=1)
    if indicators_to_plot.get("RSI Zones", False) and "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI"], mode='lines', name='RSI', line=dict(color='lightgreen')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
    if indicators_to_plot.get("MACD", False) and "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD"], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD_Signal"], mode='lines', name='MACD Signal', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["MACD_Hist"], name='MACD Hist', marker_color='grey'), row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
    if indicators_to_plot.get("Stochastic", False) and "Stoch_K" in df.columns and "Stoch_D" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["Stoch_K"], mode='lines', name='Stoch %K', line=dict(color='yellow')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["Stoch_D"], mode='lines', name='Stoch %D', line=dict(color='magenta')), row=3, col=1)
        fig.update_yaxes(title_text="Stochastic", row=3, col=1)
    fig.update_layout(
        title_text=f'{symbol} Price Chart with Indicators',
        xaxis_rangeslider_visible=False,
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["secondary_bg"],
        font=dict(color=colors["fg"]),
        xaxis=dict(gridcolor=colors["grid"]),
        yaxis=dict(gridcolor=colors["grid"]),
        height=800,
        hovermode="x unified",
        uirevision=symbol
    )
    return fig

def plot_strategy_chart(df, trades, symbol, strategy_name, chart_theme="Light", custom_colors=None):
    try:
        if custom_colors is None:
            custom_colors = {"bg": "#FFFFFF", "fg": "#000000", "grid": "rgba(128, 128, 128, 0.2)"}
        if df.empty or not trades:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candles"))
        buy_trades = [t for t in trades if t["Action"] == "Buy"]
        sell_trades = [t for t in trades if t["Action"] == "Sell"]
        fig.add_trace(go.Scatter(x=[t["Date"] for t in buy_trades], y=[t["Price"] for t in buy_trades], mode="markers", name="Buy", marker=dict(color="green", size=12, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=[t["Date"] for t in sell_trades], y=[t["Price"] for t in sell_trades], mode="markers", name="Sell", marker=dict(color="red", size=12, symbol="triangle-down")))
        bg_color = custom_colors["bg"] if chart_theme == "Custom" else "#FFFFFF" if chart_theme == "Light" else "#0E1117"
        fg_color = custom_colors["fg"] if chart_theme == "Custom" else "#000000" if chart_theme == "Light" else "#FFFFFF"
        height = 600 if st.session_state.get("is_mobile", False) else 800
        font_size = 10 if st.session_state.get("is_mobile", False) else 14
        fig.update_layout(
            title=f"{symbol} - {strategy_name} Trades",
            template="plotly_white" if chart_theme == "Light" else "plotly_dark" if chart_theme == "Dark" else None,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=fg_color, size=font_size),
            height=height,
            xaxis_rangeslider_visible=False,
        )
        return fig
    except Exception as e:
        logging.error(f"Error plotting strategy chart for {symbol}: {str(e)}\n{traceback.format_exc()}")
        return go.Figure()

def plot_correlation_matrix(dfs, symbols, chart_theme="Light", custom_colors=None):
    try:
        if len(dfs) < 2:
            return go.Figure()
        close_prices = pd.DataFrame({symbol: df["close"] for symbol, df in dfs.items()})
        corr_matrix = close_prices.corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=symbols, y=symbols, colorscale="Viridis", showscale=True))
        colors = custom_colors or {"bg": "#FFFFFF", "fg": "#000000", "grid": "rgba(128, 128, 128, 0.2)"}
        fig.update_layout(
            title="Correlation Matrix",
            template="plotly_dark" if chart_theme == "Dark" else "plotly_white",
            paper_bgcolor=colors["bg"],
            plot_bgcolor=colors["bg"],
            font=dict(color=colors["fg"]),
            height=600
        )
        return fig
    except Exception as e:
        logging.error(f"Error plotting correlation matrix: {str(e)}\n{traceback.format_exc()}")
        return go.Figure()

def plot_portfolio_performance(chart_theme="Light", custom_colors=None):
    try:
        colors = custom_colors or {"bg": "#FFFFFF", "fg": "#000000", "grid": "rgba(128, 128, 128, 0.2)"}
        portfolio = st.session_state.portfolio
        trades = portfolio["trades"]
        if not trades:
            return go.Figure()
        df_trades = pd.DataFrame(trades)
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        df_trades = df_trades.sort_values(by='exit_date')
        initial_investment = st.session_state.get("selected_investment_amount", 100.0)
        df_trades["profit_valid"] = df_trades["profit"].fillna(0)
        df_trades["cumulative_profit"] = df_trades["profit_valid"].cumsum()
        df_trades["equity"] = initial_investment + df_trades["cumulative_profit"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_trades["exit_date"], y=df_trades["equity"], name="Equity Curve", line=dict(color="blue")))
        bg_color = colors["bg"]
        fg_color = colors["fg"]
        height = 400 if st.session_state.get("is_mobile", False) else 600
        font_size = 10 if st.session_state.get("is_mobile", False) else 14
        fig.update_layout(
            title="Portfolio Equity Curve",
            template="plotly_white" if chart_theme == "Light" else "plotly_dark" if chart_theme == "Dark" else None,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=fg_color, size=font_size),
            height=height,
        )
        return fig
    except Exception as e:
        logging.error(f"Error plotting portfolio performance: {str(e)}\n{traceback.format_exc()}")
        return go.Figure()

# --- Backtesting ---
def backtest_strategy(df, strategy_name, params=None, indicators=None):
    if params is None:
        params = {}
    if indicators is None:
        indicators = {}
    if df.empty:
        return 0, 0, 0
    df_processed = df.copy()
    df_processed["returns"] = df_processed["close"].pct_change()
    df_processed["Signal"] = 0
    df_processed = add_indicators(df_processed, indicators=indicators, params=params)
    if strategy_name == "Moving Average Crossover":
        if "MA20" in df_processed.columns and "MA50" in df_processed.columns:
            df_processed.loc[df_processed["MA20"] > df_processed["MA50"], "Signal"] = 1
            df_processed.loc[df_processed["MA20"] < df_processed["MA50"], "Signal"] = -1
    elif strategy_name == "Mean Reversion":
        df_processed["Mean"] = df_processed["close"].rolling(window=params.get("mr_window", 20)).mean()
        df_processed["Std"] = df_processed["close"].rolling(window=params.get("mr_window", 20)).std()
        df_processed.loc[df_processed["close"] < (df_processed["Mean"] - params.get("mr_std", 1.5) * df_processed["Std"]), "Signal"] = 1
        df_processed.loc[df_processed["close"] > (df_processed["Mean"] + params.get("mr_std", 1.5) * df_processed["Std"]), "Signal"] = -1
    elif strategy_name == "Momentum":
        df_processed["Momentum"] = df_processed["close"].pct_change(periods=params.get("mom_period", 5))
        df_processed.loc[df_processed["Momentum"] > params.get("mom_threshold", 0.05), "Signal"] = 1
        df_processed.loc[df_processed["Momentum"] < -params.get("mom_threshold", 0.05), "Signal"] = -1
    elif strategy_name == "Bollinger Bands":
        if "BB_Low" not in df_processed.columns or "BB_High" not in df_processed.columns:
            if len(df_processed) >= params.get("bb_window", 20):
                rolling_mean = df_processed["close"].rolling(window=params.get("bb_window", 20)).mean()
                rolling_std = df_processed["close"].rolling(window=params.get("bb_window", 20)).std()
                df_processed["BB_High"] = rolling_mean + (rolling_std * params.get("bb_std", 2.0))
                df_processed["BB_Low"] = rolling_mean - (rolling_std * params.get("bb_std", 2.0))
            else:
                df_processed["BB_High"] = np.nan
                df_processed["BB_Low"] = np.nan
        df_processed.loc[df_processed["close"] < df_processed["BB_Low"], "Signal"] = 1
        df_processed.loc[df_processed["close"] > df_processed["BB_High"], "Signal"] = -1
    elif strategy_name == "RSI Divergence":
        if "RSI" not in df_processed.columns:
            if len(df_processed) >= params.get("rsi_period", 14):
                delta = df_processed["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
                rs = gain / loss
                df_processed["RSI"] = 100 - (100 / (1 + rs))
            else:
                df_processed["RSI"] = np.nan
        df_processed["Price_Diff"] = df_processed["close"].diff()
        df_processed["RSI_Diff"] = df_processed["RSI"].diff()
        df_processed["Signal"] = 0
        mask_buy = (df_processed["Price_Diff"] > 0) & (df_processed["RSI_Diff"] < 0) & (df_processed["RSI"] < params.get("alert_rsi_oversold", 30.0))
        mask_sell = (df_processed["Price_Diff"] < 0) & (df_processed["RSI_Diff"] > 0) & (df_processed["RSI"] > params.get("alert_rsi_overbought", 70.0))
        df_processed.loc[mask_buy, "Signal"] = 1
        df_processed.loc[mask_sell, "Signal"] = -1
    elif strategy_name == "Combined Indicators":
        signals_columns = []
        if indicators.get("MA20/50", False) and "MA20" in df_processed.columns and "MA50" in df_processed.columns:
            df_processed["MA_Signal"] = 0
            df_processed.loc[df_processed["MA20"] > df_processed["MA50"], "MA_Signal"] = 1
            df_processed.loc[df_processed["MA20"] < df_processed["MA50"], "MA_Signal"] = -1
            signals_columns.append("MA_Signal")
        if indicators.get("Bollinger Bands", False) and "BB_Low" in df_processed.columns and "BB_High" in df_processed.columns:
            df_processed["BB_Signal"] = 0
            df_processed.loc[df_processed["close"] < df_processed["BB_Low"], "BB_Signal"] = 1
            df_processed.loc[df_processed["close"] > df_processed["BB_High"], "BB_Signal"] = -1
            signals_columns.append("BB_Signal")
        if indicators.get("RSI Zones", False) and "RSI" in df_processed.columns:
            df_processed["RSI_Signal"] = 0
            df_processed.loc[df_processed["RSI"] < params.get("alert_rsi_oversold", 30.0), "RSI_Signal"] = 1
            df_processed.loc[df_processed["RSI"] > params.get("alert_rsi_overbought", 70.0), "RSI_Signal"] = -1
            signals_columns.append("RSI_Signal")
        if indicators.get("MACD", False) and "MACD" in df_processed.columns and "MACD_Signal" in df_processed.columns:
            df_processed["MACD_Signal_Combined"] = 0
            df_processed.loc[(df_processed["MACD"] > df_processed["MACD_Signal"]) & (df_processed["MACD"].shift(1) <= df_processed["MACD_Signal"].shift(1)), "MACD_Signal_Combined"] = 1
            df_processed.loc[(df_processed["MACD"] < df_processed["MACD_Signal"]) & (df_processed["MACD"].shift(1) >= df_processed["MACD_Signal"].shift(1)), "MACD_Signal_Combined"] = -1
            signals_columns.append("MACD_Signal_Combined")
        if indicators.get("Stochastic", False) and "Stoch_K" in df_processed.columns and "Stoch_D" in df_processed.columns:
            df_processed["Stoch_Signal"] = 0
            df_processed.loc[(df_processed["Stoch_K"] < 20) & (df_processed["Stoch_K"] > df_processed["Stoch_D"]), "Stoch_Signal"] = 1
            df_processed.loc[(df_processed["Stoch_K"] > 80) & (df_processed["Stoch_K"] < df_processed["Stoch_D"]), "Stoch_Signal"] = -1
            signals_columns.append("Stoch_Signal")
        if indicators.get("ATR", False) and "ATR" in df_processed.columns:
            df_processed["ATR_Signal"] = 0
            atr_threshold = df_processed["ATR"].mean() * 1.5 if not df_processed["ATR"].isnull().all() else 0
            df_processed.loc[df_processed["close"].diff() > atr_threshold, "ATR_Signal"] = 1
            df_processed.loc[df_processed["close"].diff() < -atr_threshold, "ATR_Signal"] = -1
            signals_columns.append("ATR_Signal")
        if signals_columns:
            df_processed["Consensus_Signal"] = df_processed[signals_columns].mean(axis=1)
            df_processed.loc[df_processed["Consensus_Signal"] >= 0.6, "Signal"] = 1
            df_processed.loc[df_processed["Consensus_Signal"] <= -0.6, "Signal"] = -1
        else:
            return 0, 0, 0
    df_processed["strategy_returns"] = df_processed["Signal"].shift(1) * df_processed["returns"]
    df_processed.dropna(subset=["strategy_returns"], inplace=True)
    if df_processed.empty:
        return 0, 0, 0
    sharpe_ratio = (
        np.sqrt(252) * df_processed["strategy_returns"].mean() / df_processed["strategy_returns"].std()
        if df_processed["strategy_returns"].std() != 0
        else 0
    )
    total_return = (df_processed["strategy_returns"] + 1).prod() - 1
    cumulative_returns = (1 + df_processed["strategy_returns"]).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    return sharpe_ratio, total_return, max_drawdown

# --- AI Assistant Logic ---
def ai_assistant_response(user_query, current_data):
    query_lower = user_query.lower()
    if "hello" in query_lower or "hi" in query_lower:
        return "Hello! How can I assist you with crypto market analysis today?"
    elif "price of" in query_lower:
        symbol_query = query_lower.split("price of")[-1].strip().upper().replace(" ", "")
        current_price = current_data.get("current_price")
        if current_price and symbol_query in current_data.get("symbol", ""):
            return f"The current price of {current_data['symbol']} is ${current_price:.2f}."
        else:
            return f"I can't find the current price for {symbol_query}. Please ensure it's selected in the sidebar and you've run the analysis."
    elif "predict" in query_lower and "price" in query_lower:
        predictions = current_data.get("predictions")
        symbol = current_data.get("symbol")
        if predictions is not None and len(predictions) > 0 and symbol:
            return f"The model predicts the price of {symbol} to be around ${predictions[-1]:.2f} in the next few periods. Check the Dashboard tab for the full prediction chart!"
        else:
            return "I need to run a prediction first! Please go to the Dashboard tab and run the analysis."
    elif "trend" in query_lower:
        trend = current_data.get("trend")
        if trend:
            return f"The current market trend is **{trend}**."
        else:
            return "I'm not sure about the current trend. Please ensure data is loaded and analysis is run."
    elif "rsi" in query_lower:
        return "The Relative Strength Index (RSI) is a momentum indicator that measures the speed and change of price movements. It helps identify overbought (above 70) or oversold (below 30) conditions."
    elif "bollinger bands" in query_lower or "bb" in query_lower:
        return "Bollinger Bands measure market volatility. They consist of a middle band (SMA) and two outer bands that expand and contract with volatility."
    elif "macd" in query_lower:
        return "MACD shows the relationship between two moving averages of a security’s price. It helps identify buy and sell signals."
    elif "how to use this app" in query_lower or "help" in query_lower:
        return "You can configure symbols, timeframes, indicators, and models in the sidebar. Explore tabs like Dashboard for live data, Simulation for virtual trading, and Documentation for detailed info!"
    elif "strategies" in query_lower:
        return "We offer strategies like Moving Average Crossover, Mean Reversion, Momentum, Bollinger Bands, RSI Divergence, and a Combined Indicators strategy."
    elif "what is" in query_lower:
        return "I'm still learning to understand complex 'what is' queries. For now, try simple questions about market data, predictions, or indicators!"
    else:
        return f"I'm still learning to understand complex queries like '{user_query}'. For now, try simple questions about market data, predictions, or indicators!"

# --- MAIN STREAMLIT APP LAYOUT ---
async def main():
    st.title("AI Crypto/Stock Predictor")
    if 'user_agent' not in st.session_state:
        st.session_state.user_id = st.session_state.get("user_id", "default_user")
        try:
            st.session_state.user_agent = st.query_params.get('user_agent', [''])[0]
        except AttributeError:
            st.session_state.user_agent = ''
            logging.warning("Streamlit version does not support query_params. Using empty user_agent.")
        st.session_state.is_mobile = any(m in st.session_state.user_agent.lower() for m in ['android', 'iphone', 'ipad'])

    # Initialize session state
    defaults = {
        "db": None, "db_type": "None", "user_id": "default_user",
        "selected_exchange_name": "Binance", "selected_symbols": ["BTC/USDT"],
        "selected_timeframe": "1h",
        "selected_indicators_map": {
            "MA20/50": True, "Bollinger Bands": True, "RSI Zones": True,
            "MACD": True, "VWAP": True, "Stochastic": True, "ATR": True, "Momentum": True
        },
        "indicator_params": {
            "ma_short": 20, "ma_long": 50, "rsi_period": 14, "alert_rsi_oversold": 30,
            "alert_rsi_overbought": 70, "macd_fast": 12, "macd_slow": 26,
            "macd_signal": 9, "bb_window": 20, "bb_std": 2.0,
            "mom_period": 5, "mom_threshold": 0.05, "mr_window": 20, "mr_std": 1.5
        },
        "selected_model": "Ensemble",
        "selected_strategies_map": {
            "Moving Average Crossover": True, "Mean Reversion": True, "Momentum": True,
            "Bollinger Bands": True, "RSI Divergence": True, "Combined Indicators": True
        },
        "selected_investment_amount": 1000,
        "news_api_key": "671179e3e1114aeaa9950d6930176d46",
        "prediction_horizon": 15,
        "training_epochs": 50,
        "tuning_epochs": 10,
        "refresh_interval": 0,
        "current_data": {},
        "portfolio": {"positions": {}, "trades": [], "balance": 1000},
        "strategy_results": {},
        "lstm_model": None, "transformer_model": None, "arima_model": None, "linear_model": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.button(f"Switch to {'Light' if st.session_state.theme == 'Dark' else 'Dark'} Theme", on_click=toggle_theme)
        db_options = ["None", "MongoDB", "MSSQL", "MySQL"]
        st.selectbox("Database Type", db_options, index=db_options.index(st.session_state.db_type), key="db_type")
        st.text_input("User ID", st.session_state.user_id, key="user_id")
        if st.session_state.db_type != "None":
            st.info("Enter database credentials. Click 'Run Analysis' to connect and save.")
            st.text_input("DB Host", "localhost", key="db_host")
            st.text_input("DB Port", "27017" if st.session_state.db_type == "MongoDB" else "1433" if st.session_state.db_type == "MSSQL" else "3306", key="db_port")
            st.text_input("DB Username", "user", key="db_username")
            st.text_input("DB Password", "password", type="password", key="db_password")
            st.text_input("DB Name", "CryptoPredictor", key="db_name")
        st.subheader("📈 Market Data Settings")
        exchanges_list = list(EXCHANGES.keys())
        st.selectbox("Select Exchange", exchanges_list, index=exchanges_list.index(st.session_state.selected_exchange_name) if st.session_state.selected_exchange_name in exchanges_list else 0, key="selected_exchange_name")
        @st.cache_data(ttl=600)
        def fetch_markets_cached(exchange_name):
            async def fetch_markets_async(exchange_name):
                exchange_constructor = EXCHANGES.get(exchange_name)
                if not exchange_constructor:
                    return []
                exchange = exchange_constructor()
                try:
                    await exchange.load_markets()
                    markets = exchange.markets
                    symbols = [market for market in markets if markets[market].get("active", False)]
                    formatted_symbols = []
                    for symbol in symbols:
                        market_info = markets[symbol]
                        base = market_info.get('base', '')
                        quote = market_info.get('quote', '')
                        if base and quote:
                            formatted_symbols.append(f"{base}/{quote}")
                        else:
                            formatted_symbols.append(symbol)
                    return sorted(formatted_symbols)
                except Exception as e:
                    logging.error(f"Failed to fetch markets for {exchange_name}: {str(e)}")
                    return ['BTC/USDT']
                finally:
                    try:
                        await exchange.close()
                    except Exception:
                        pass
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(fetch_markets_async(exchange_name))
            finally:
                loop.close()
        symbol_options = fetch_markets_cached(st.session_state.selected_exchange_name)
        st.multiselect("Select Symbols", symbol_options, default=st.session_state.selected_symbols, key="selected_symbols")
        timeframes = ["1m", "2m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M", "3M"]
        st.selectbox("Select Timeframe", timeframes, index=timeframes.index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in timeframes else 6, key="selected_timeframe")
        st.subheader("📊 Technical Indicators")
        indicator_options = {
            "MA20/50": "Moving Averages (20, 50)", "Bollinger Bands": "Bollinger Bands",
            "RSI Zones": "RSI Zones", "MACD": "MACD", "VWAP": "VWAP",
            "Stochastic": "Stochastic", "ATR": "ATR", "Momentum": "Momentum"
        }
        for k, v in indicator_options.items():
            st.session_state.selected_indicators_map[k] = st.checkbox(v, value=st.session_state.selected_indicators_map.get(k, True), key=f"ind_{k}")
        with st.expander("Indicator Parameters"):
            st.session_state.indicator_params["ma_short"] = st.slider("MA Short Period", 5, 50, value=st.session_state.indicator_params.get("ma_short", 20))
            st.session_state.indicator_params["ma_long"] = st.slider("MA Long Period", 20, 200, value=st.session_state.indicator_params.get("ma_long", 50))
            st.session_state.indicator_params["bb_window"] = st.slider("BB Window", 10, 50, value=st.session_state.indicator_params.get("bb_window", 20))
            st.session_state.indicator_params["bb_std"] = st.slider("BB Std Dev", 1.0, 3.0, value=st.session_state.indicator_params.get("bb_std", 2.0), step=0.1)
            st.session_state.indicator_params["rsi_period"] = st.slider("RSI Period", 5, 20, value=st.session_state.indicator_params.get("rsi_period", 14))
            st.session_state.indicator_params["alert_rsi_oversold"] = st.slider("RSI Oversold", 20, 40, value=st.session_state.indicator_params.get("alert_rsi_oversold", 30))
            st.session_state.indicator_params["alert_rsi_overbought"] = st.slider("RSI Overbought", 60, 80, value=st.session_state.indicator_params.get("alert_rsi_overbought", 70))
            st.session_state.indicator_params["macd_fast"] = st.slider("MACD Fast Period", 5, 20, value=st.session_state.indicator_params.get("macd_fast", 12))
            st.session_state.indicator_params["macd_slow"] = st.slider("MACD Slow Period", 20, 50, value=st.session_state.indicator_params.get("macd_slow", 26))
            st.session_state.indicator_params["macd_signal"] = st.slider("MACD Signal Period", 5, 15, value=st.session_state.indicator_params.get("macd_signal", 9))
            st.session_state.indicator_params["mom_period"] = st.slider("Momentum Period", 3, 10, value=st.session_state.indicator_params.get("mom_period", 5))
            st.session_state.indicator_params["mom_threshold"] = st.slider("Momentum Threshold", 0.01, 0.1, value=st.session_state.indicator_params.get("mom_threshold", 0.05), step=0.01)
            st.session_state.indicator_params["mr_window"] = st.slider("Mean Reversion Window", 10, 30, value=st.session_state.indicator_params.get("mr_window", 20))
            st.session_state.indicator_params["mr_std"] = st.slider("Mean Reversion Std Dev", 1.0, 3.0, value=st.session_state.indicator_params.get("mr_std", 1.5), step=0.1)
        st.subheader("🤖 Prediction Models")
        model_options = ["LSTM", "Transformer", "ARIMA", "Linear Regression", "Ensemble"]
        st.selectbox("Select Prediction Model", model_options, index=model_options.index(st.session_state.selected_model), key="selected_model")
        st.slider("Prediction Horizon (periods)", 5, 60, value=st.session_state.prediction_horizon, key="prediction_horizon")
        st.slider("Training Epochs (for ML models)", 10, 100, value=st.session_state.training_epochs, key="training_epochs")
        st.slider("Hyperparameter Tuning Epochs", 5, 20, value=st.session_state.tuning_epochs, key="tuning_epochs")
        st.subheader("⚡️ Trading Strategies")
        strategy_options = {
            "Moving Average Crossover": "Moving Average Crossover", "Mean Reversion": "Mean Reversion",
            "Momentum": "Momentum", "Bollinger Bands": "Bollinger Bands",
            "RSI Divergence": "RSI Divergence", "Combined Indicators": "Combined Indicators"
        }
        for k, v in strategy_options.items():
            st.session_state.selected_strategies_map[k] = st.checkbox(v, value=st.session_state.selected_strategies_map.get(k, True), key=f"strategy_checkbox_{k}")
        st.subheader("💰 Simulation Settings")
        st.slider("Initial Investment ($)", 100, 10000, value=st.session_state.selected_investment_amount, key="selected_investment_amount")
        st.subheader("📰 NewsAPI Key")
        st.text_input("NewsAPI Key (for Sentiment Analysis)", st.session_state.news_api_key, type="password", key="news_api_key")
        st.subheader("🔄 Auto-Refresh")
        st.number_input("Auto-refresh interval (seconds, 0 for off)", min_value=0, value=st.session_state.refresh_interval, key="refresh_interval")
        if st.session_state.refresh_interval > 0:
            st.caption(f"Page will refresh every {st.session_state.refresh_interval} seconds.")
        if st.button("🚀 Run Analysis & Save Preferences", key="run_analysis_button"):
            if st.session_state.db_type != "None":
                try:
                    st.session_state.db = init_db(
                        st.session_state.db_type,
                        st.session_state.db_host,
                        st.session_state.db_port,
                        st.session_state.db_username,
                        st.session_state.db_password,
                        st.session_state.db_name
                    )
                    if st.session_state.db:
                        st.sidebar.success("Database connected successfully!")
                        save_preferences(
                            st.session_state.db,
                            st.session_state.db_type,
                            st.session_state.user_id,
                            st.session_state.selected_exchange_name,
                            st.session_state.selected_symbols,
                            st.session_state.selected_timeframe,
                            st.session_state.selected_indicators_map,
                            st.session_state.selected_strategies_map,
                            st.session_state.news_api_key
                        )
                        st.sidebar.success("Preferences saved to database!")
                    else:
                        st.sidebar.error("Failed to connect to database. Running without database persistence.")
                except Exception as e:
                    st.sidebar.error(f"Error connecting to database: {e}. Running without database persistence.")
                    st.session_state.db = None
            else:
                st.sidebar.info("No database selected. Running without persistence.")
            st.rerun()

    # Main content
    tabs = st.tabs(["Dashboard", "Model Performance", "Order Book", "Multi-Exchange", "Simulation & Strategies", "Backtest", "Detailed Predictions", "Comparison", "💬 AI Assistant", "📰 News", "Documentation"])
    
    with tabs[0]:
        st.subheader("Live Market Data & Predictions")
        if not st.session_state.selected_symbols:
            st.info("Please select at least one symbol to display data in the sidebar and click 'Run Analysis'.")
        else:
            for symbol in st.session_state.selected_symbols:
                with st.container():
                    st.markdown(f"### {symbol} Analysis")
                    try:
                        ohlcv_df = get_ohlcv(st.session_state.selected_exchange_name, symbol, st.session_state.selected_timeframe, limit=200)
                        if ohlcv_df.empty:
                            st.warning(f"No data available for {symbol}.")
                            continue
                        ohlcv_df["symbol"] = symbol
                        funding_rate, order_book_imbalance, sentiment_score = await fetch_external_data(st.session_state.selected_exchange_name, symbol, st.session_state.news_api_key)
                        ohlcv_df["funding_rate"] = funding_rate
                        ohlcv_df["order_book_imbalance"] = order_book_imbalance
                        ohlcv_df["sentiment_score"] = sentiment_score
                        processed_df = add_indicators(ohlcv_df, indicators=st.session_state.selected_indicators_map, params=st.session_state.indicator_params)
                        current_price = processed_df["close"].iloc[-1]
                        trend = detect_trend(processed_df)
                        support, resistance = support_resistance(processed_df)
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric(label=f"Current Price ({symbol})", value=f"${current_price:,.2f}")
                        col2.metric("Trend", trend)
                        col3.metric("Support", f"${support:,.2f}" if support else "N/A")
                        col4.metric("Resistance", f"${resistance:,.2f}" if resistance else "N/A")
                        col5.metric("Sentiment Score", f"{sentiment_score:.2f}")
                        with st.spinner(f"Running {st.session_state.selected_model} prediction for {symbol}..."):
                            predictions, trained_model, rmse, mae, error_msg = train_and_predict_model(
                                st.session_state.selected_model,
                                processed_df.copy(),
                                n_future=st.session_state.prediction_horizon,
                                epochs=st.session_state.training_epochs,
                                batch_size=32,
                                tuning_epochs=st.session_state.tuning_epochs
                            )
                        if error_msg:
                            st.error(f"Prediction error for {symbol}: {error_msg}")
                            predictions = np.array([])
                        else:
                            st.session_state[f"predictions_{symbol}"] = predictions
                            st.session_state[f"rmse_{symbol}"] = rmse
                            st.session_state[f"mae_{symbol}"] = mae
                            st.session_state.current_data = {
                                "symbol": symbol,
                                "current_price": current_price,
                                "trend": trend,
                                "predictions": predictions.tolist()
                            }
                            volatility = processed_df['close'].pct_change().std()
                            confidence_interval = np.array([volatility * np.sqrt(i) * current_price for i in range(1, len(predictions) + 1)]) * 1.96 if pd.notna(volatility) and np.isfinite(volatility) else np.zeros_like(predictions)
                            st.session_state[f"confidence_interval_{symbol}"] = confidence_interval
                            fig = plot_candlestick(processed_df, symbol, st.session_state.selected_indicators_map, predictions, confidence_interval, st.session_state.theme, colors)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        st.subheader("Educational Insights")
                        expander_cols = st.columns(2)
                        with expander_cols[0]:
                            with st.expander("What is RSI?"):
                                st.markdown("The Relative Strength Index (RSI) is a momentum indicator that measures the speed and magnitude of recent price changes to evaluate overvalued or undervalued conditions in the price of a stock or other asset.")
                        with expander_cols[1]:
                            with st.expander("What are Bollinger Bands?"):
                                st.markdown("Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity. They consist of a middle band (a simple moving average) and two outer bands (upper and lower bands) that are typically two standard deviations away from the middle band.")
                        with st.expander("Example Signals Explained"):
                            st.markdown("""
                            * **RSI Oversold (RSI < 30):** Indicates the asset may be undervalued and due for a price correction upwards.
                            * **RSI Overbought (RSI > 70):** Indicates the asset may be overvalued and due for a price correction downwards.
                            * **MA Crossover (MA20 > MA50):** A bullish signal, suggesting a potential uptrend.
                            * **MA Crossover (MA20 < MA50):** A bearish signal, suggesting a potential downtrend.
                            """)
                    except Exception as e:
                        st.error(f"Error processing {symbol}: {str(e)}\n{traceback.format_exc()}")

    with tabs[1]:
        st.subheader("AI/ML Model Performance Showcase")
        if not st.session_state.selected_symbols:
            st.info("Please select symbols to view model performance in the sidebar and click 'Run Analysis'.")
        else:
            for symbol in st.session_state.selected_symbols:
                st.write(f"### {symbol} Model Metrics")
                rmse = st.session_state.get(f"rmse_{symbol}", None)
                mae = st.session_state.get(f"mae_{symbol}", None)
                if rmse is not None and mae is not None:
                    col_rmse, col_mae, col_sharpe = st.columns(3)
                    with col_rmse:
                        st.metric("RMSE (scaled)", f"{rmse:.4f}")
                    with col_mae:
                        st.metric("MAE (scaled)", f"{mae:.4f}")
                    with col_sharpe:
                        total_return, sharpe_ratio, max_drawdown, equity_curve = calculate_portfolio_metrics()
                        st.metric("Sharpe Ratio (Simulated)", f"{sharpe_ratio:.2f}")
                    st.subheader("Confidence Intervals for Predictions")
                    predictions = st.session_state.get(f"predictions_{symbol}", None)
                    confidence_interval = st.session_state.get(f"confidence_interval_{symbol}", None)
                    if predictions is not None and confidence_interval is not None and len(predictions) > 0:
                        freq_map = {"1m": "T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "H", "4h": "4H", "1d": "D"}
                        freq = freq_map.get(st.session_state.selected_timeframe, "T")
                        pred_timestamps = pd.date_range(pd.Timestamp.now(), periods=len(predictions), freq=freq)
                        fig_ci = go.Figure()
                        fig_ci.add_trace(go.Scatter(x=pred_timestamps, y=predictions, mode='lines', name='Predicted Price', line=dict(color='blue')))
                        fig_ci.add_trace(go.Scatter(x=pred_timestamps, y=predictions + confidence_interval, mode='lines', line=dict(width=0), name='Upper Bound', showlegend=False))
                        fig_ci.add_trace(go.Scatter(x=pred_timestamps, y=predictions - confidence_interval, mode='lines', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', name='Confidence Interval', showlegend=False))
                        fig_ci.update_layout(title=f'{symbol} Price Prediction with Confidence Interval',
                                             xaxis_title="Time", yaxis_title="Price",
                                             template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white",
                                             paper_bgcolor=colors["bg"], plot_bgcolor=colors["secondary_bg"],
                                             font=dict(color=colors["fg"]), xaxis=dict(gridcolor=colors["grid"]), yaxis=dict(gridcolor=colors["grid"]))
                        st.plotly_chart(fig_ci, use_container_width=True)
                    else:
                        st.info("Run predictions on the Dashboard tab to see confidence intervals.")
                    st.subheader("Explainable AI (XAI) Insights")
                    st.info("This section would display SHAP or LIME plots to explain how input features (e.g., RSI, Volume) contribute to the model's prediction.")
                else:
                    st.info(f"No performance metrics available for {symbol}. Please ensure data is loaded and model is run.")

    with tabs[2]:
        st.subheader("Real-Time Order Book Depth & Imbalance")
        if not st.session_state.selected_symbols:
            st.info("Please select a symbol to view order book data.")
        else:
            symbol = st.selectbox("Select a symbol for Order Book", st.session_state.selected_symbols)
            if symbol:
                depth = st.slider(f"Order Book Depth ({symbol})", 5, 50, value=20, key=f"{symbol}_order_book_depth")
                bids_df, asks_df, spread, imbalance = await fetch_order_book(st.session_state.selected_exchange_name, symbol, depth)
                if not bids_df.empty and not asks_df.empty:
                    st.metric("Bid-Ask Spread", f"{spread:.4f}" if spread is not None else "N/A")
                    st.metric("Order Book Imbalance", f"{imbalance:.2f}" if imbalance is not None else "N/A")
                    fig_order_book = plot_order_book(bids_df, asks_df, symbol, st.session_state.theme, colors)
                    st.plotly_chart(fig_order_book, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.warning(f"Could not fetch order book data for {symbol}.")

    with tabs[3]:
        st.subheader("Multi-Exchange Symbol Availability & Metrics")
        comparison_exchanges = ["Binance", "Kraken", "Bybit", "MEXC", "Bitget", "BingX"]
        comparison_data = []
        for ex_name in comparison_exchanges:
            temp_exchange_constructor = EXCHANGES.get(ex_name)
            if not temp_exchange_constructor: continue
            temp_exchange = temp_exchange_constructor()
            try:
                await temp_exchange.load_markets()
                for symbol in st.session_state.selected_symbols:
                    is_available = any(s for s in temp_exchange.markets if s.startswith(symbol.replace('/', ''))) or symbol in temp_exchange.markets
                    if is_available:
                        comparison_data.append({"Exchange": ex_name, "Symbol": symbol, "Available": "✅ Yes"})
                    else:
                        comparison_data.append({"Exchange": ex_name, "Symbol": symbol, "Available": "❌ No"})
            except Exception as e:
                logging.warning(f"Could not load markets for {ex_name}: {e}")
            finally:
                if temp_exchange: await temp_exchange.close()
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data))
        else:
            st.info("No comparison data available.")

    with tabs[4]:
        st.subheader("Virtual Auto-Trading Bot Simulation & Strategy Details")
        init_portfolio()
        selected_strategies = [strategy for strategy, enabled in st.session_state.selected_strategies_map.items() if enabled]
        if st.button("Run Virtual Trading Simulation", key="run_simulation_button"):
            if not st.session_state.selected_symbols or not selected_strategies:
                st.warning("Please select at least one symbol and strategy.")
            else:
                for symbol in st.session_state.selected_symbols:
                    ohlcv_df = get_ohlcv(st.session_state.selected_exchange_name, symbol, st.session_state.selected_timeframe, limit=200)
                    if ohlcv_df.empty:
                        st.warning(f"No data for {symbol}.")
                        continue
                    processed_df = add_indicators(ohlcv_df, indicators=st.session_state.selected_indicators_map, params=st.session_state.indicator_params)
                    predictions = st.session_state.get(f"predictions_{symbol}", np.array([]))
                    if len(predictions) == 0:
                        st.warning(f"No predictions available for {symbol}. Please run predictions on the Dashboard tab first.")
                        continue
                    freq_map = {"1m": "T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "H", "4h": "4H", "1d": "D"}
                    freq = freq_map.get(st.session_state.selected_timeframe, "T")
                    start_timestamp_for_preds = ohlcv_df["timestamp"].iloc[-1] if not ohlcv_df.empty else pd.Timestamp.now()
                    future_x = pd.date_range(start_timestamp_for_preds, periods=len(predictions) + 1, freq=freq)[1:]
                    pred_open = [ohlcv_df["close"].iloc[-1]] + list(predictions[:-1]) if not ohlcv_df.empty else [predictions[0]] + list(predictions[:-1])
                    pred_high = [p + p * 0.01 for p in predictions]
                    pred_low = [p - p * 0.01 for p in predictions]
                    pred_volume = [ohlcv_df["volume"].mean()] * len(predictions) if not ohlcv_df.empty else [1000] * len(predictions)
                    pred_candles = pd.DataFrame({
                        "timestamp": future_x,
                        "open": pred_open,
                        "high": pred_high,
                        "low": pred_low,
                        "close": predictions,
                        "volume": pred_volume,
                        "symbol": symbol
                    })
                    combined_df = pd.concat([processed_df, pred_candles], ignore_index=True)
                    combined_df_with_indicators = add_indicators(combined_df, indicators=st.session_state.selected_indicators_map, params=st.session_state.indicator_params)
                    strategy_functions = {
                        "Moving Average Crossover": moving_average_crossover,
                        "Mean Reversion": mean_reversion,
                        "Momentum": momentum_trading,
                        "Bollinger Bands": bollinger_bands_strategy,
                        "RSI Divergence": rsi_divergence_strategy,
                        "Combined Indicators": combined_indicators_strategy
                    }
                    for strategy_name in selected_strategies:
                        strategy_func = strategy_functions.get(strategy_name)
                        if strategy_func:
                            if strategy_name == "Combined Indicators":
                                trades = strategy_func(combined_df_with_indicators, st.session_state.selected_investment_amount, params=st.session_state.indicator_params, indicators=st.session_state.selected_indicators_map)
                            else:
                                trades = strategy_func(combined_df_with_indicators, st.session_state.selected_investment_amount, params=st.session_state.indicator_params)
                            if trades:
                                with st.expander(f"Trade Log for {symbol} - {strategy_name}"):
                                    trade_df = pd.DataFrame(trades)
                                    trade_df['Price'] = pd.to_numeric(trade_df['Price'], errors='coerce')
                                    trade_df['Entry_Price'] = pd.to_numeric(trade_df['Entry_Price'], errors='coerce')
                                    trade_df["Profit"] = trade_df.apply(
                                        lambda row: (row["Price"] - row["Entry_Price"]) * row["Shares"] if row["Action"] == "Sell" else None, axis=1
                                    )
                                    trade_df["Profit_Pct"] = trade_df.apply(
                                        lambda row: (row["Price"] / row["Entry_Price"] - 1) * 100 if row["Action"] == "Sell" and row["Entry_Price"] != 0 else None, axis=1
                                    )
                                    st.dataframe(
                                        trade_df[[
                                            "Date", "Action", "Price", "Entry_Price", "Shares",
                                            "Profit", "Profit_Pct", "Strategy"
                                        ]].style.format({
                                            "Price": "{:.2f}",
                                            "Entry_Price": "{:.2f}",
                                            "Shares": "{:.4f}",
                                            "Profit": "{:.2f}",
                                            "Profit_Pct": "{:.2f}%"
                                        }, na_rep="N/A")
                                    )
                                    fig = plot_strategy_chart(combined_df_with_indicators, trades, symbol, strategy_name, st.session_state.theme, colors)
                                    st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.header("🔍 Backtest Results")
        if not st.session_state.selected_symbols:
            st.info("Please select symbols to run backtests.")
        else:
            for symbol in st.session_state.selected_symbols:
                with st.expander(f"{symbol} Backtest", expanded=True):
                    df = get_ohlcv(st.session_state.selected_exchange_name, symbol, st.session_state.selected_timeframe, limit=500)
                    if df.empty:
                        st.warning(f"No data for {symbol}.")
                        continue
                    df["symbol"] = symbol
                    all_strategy_names = list(st.session_state.selected_strategies_map.keys())
                    for strategy_name in all_strategy_names:
                        if st.session_state.selected_strategies_map.get(strategy_name, False):
                            st.markdown(f"#### {strategy_name} Performance")
                            sharpe, total_return, max_drawdown = backtest_strategy(df.copy(), strategy_name, st.session_state.indicator_params, st.session_state.selected_indicators_map)
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            col2.metric("Total Return", f"{total_return * 100:.2f}%")
                            col3.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")

    with tabs[6]:
        st.header("🔮 Detailed Predictions")
        if not st.session_state.selected_symbols:
            st.info("Please select symbols to view detailed predictions.")
        else:
            for symbol in st.session_state.selected_symbols:
                with st.expander(f"{symbol} Detailed Predictions", expanded=True):
                    predictions = st.session_state.get(f"predictions_{symbol}", np.array([]))
                    ohlcv_df = get_ohlcv(st.session_state.selected_exchange_name, symbol, st.session_state.selected_timeframe, limit=200)
                    if ohlcv_df.empty:
                        st.warning(f"No data for {symbol}.")
                        continue
                    if len(predictions) > 0:
                        st.markdown(f"**🔮 Next {len(predictions)} Predicted Candles**")
                        freq_map = {"1m": "T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "H", "4h": "4H", "1d": "D"}
                        freq = freq_map.get(st.session_state.selected_timeframe, "T")
                        start_timestamp = ohlcv_df["timestamp"].iloc[-1] if not ohlcv_df.empty else pd.Timestamp.now()
                        pred_timestamps = pd.date_range(start_timestamp, periods=len(predictions) + 1, freq=freq)[1:]
                        pred_open = [ohlcv_df["close"].iloc[-1]] + list(predictions[:-1]) if not ohlcv_df.empty else [predictions[0]] + list(predictions[:-1])
                        pred_high = [p + p * 0.01 for p in predictions]
                        pred_low = [p - p * 0.01 for p in predictions]
                        pred_volume = [ohlcv_df["volume"].mean()] * len(predictions) if not ohlcv_df.empty else [1000] * len(predictions)
                        pred_df = pd.DataFrame({
                            "timestamp": pred_timestamps,
                            "open": pred_open,
                            "high": pred_high,
                            "low": pred_low,
                            "close": predictions,
                            "volume": pred_volume
                        })
                        st.dataframe(
                            pred_df[["timestamp", "open", "high", "low", "close", "volume"]].style.format({
                                "open": "{:.2f}",
                                "high": "{:.2f}",
                                "low": "{:.2f}",
                                "close": "{:.2f}",
                                "volume": "{:.0f}"
                            })
                        )
                        st.markdown("**📈 Predictions Summary**")
                        for i, val in enumerate(predictions, 1):
                            pct = (val - ohlcv_df["close"].iloc[-1]) / ohlcv_df["close"].iloc[-1] * 100
                            st.write(f"• Prediction T+{i}: `${val:,.2f}` ({'🟢' if pct >= 0 else '🔴'} {pct:.2f}%)")
                        tv_interval_map = {
                            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
                            "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1M": "M"
                        }
                        tv_interval = tv_interval_map.get(st.session_state.selected_timeframe, "60")
                        tv_symbol = symbol.replace("/", "")
                        theme_mode = st.session_state.theme.lower() if st.session_state.theme in ["Light", "Dark"] else "light"
                        st.components.v1.html(
                            f"""
                            <iframe src="https://s.tradingview.com/widgetembed/?symbol={st.session_state.selected_exchange_name.upper()}:{tv_symbol.upper()}&interval={tv_interval}&theme={theme_mode}&style=1"
                                    width="100%" height="600" frameborder="0"></iframe>
                            """, height=600
                        )
                    else:
                        st.info(f"No predictions available for {symbol}. Run analysis from the Dashboard.")

    with tabs[7]:
        st.header("🔍 Strategy Comparison")
        st.info("This tab is a placeholder for comparing saved predictions against actual market data.")

    with tabs[8]:
        st.header("💬 JARVIS-like AI Assistant")
        st.info("Note: Voice input requires installing 'speech_recognition' and a microphone setup.")
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze the market?"}]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if user_query := st.chat_input("Ask about trends, indicators, or predictions..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.spinner("Thinking..."):
                response = ai_assistant_response(user_query, st.session_state.get("current_data", {}))
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if "male" in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                    engine.say(response)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    st.warning(f"Could not play audio response: {str(e)}")

    with tabs[9]:
        st.subheader("📰 Latest Crypto News")
        if not st.session_state.news_api_key:
            st.warning("Please provide a NewsAPI Key in the sidebar.")
        else:
            query = " OR ".join(s.split('/')[0] for s in st.session_state.selected_symbols)
            news_articles = search_x(query, st.session_state.news_api_key).get("posts", [])
            if not news_articles:
                st.info("No news found.")
            else:
                for article in news_articles:
                    title = article.get('title', 'No Title')
                    url = article.get('url', '#')
                    text = article.get('text', '')
                    source = article.get('source', 'Unknown')
                    published_at = article.get('publishedAt', '')
                    st.markdown(f"#### [{title}]({url})")
                    st.markdown(f"*{text}*")
                    st.caption(f"{source} — {published_at}")

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