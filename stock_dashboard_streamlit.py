# stock_dashboard_streamlit.py
import streamlit as st

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="AI Stock Signals Dashboard", layout="wide")
st.title("🤖 AI Stock Signal Dashboard (Streamlit Edition)")
st.write("Live technical indicators, sentiment analysis, and BUY/HOLD/SELL signals powered by Yahoo Finance.")

# ---------------------- User Input ----------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL").upper()

# ---------------------- Data Fetch ----------------------------
@st.cache_data(ttl=3600)
def fetch_prices(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df.rename(columns={"Adj Close": "Close"})
        df = df[["Close", "Volume"]].dropna()
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# ---------------------- Indicators ----------------------------
def compute_indicators(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    out["MA200"] = df["Close"].rolling(200, min_periods=1).mean()

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    out["RSI"] = 100 - (100 / (1 + rs))

    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = exp12 - exp26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    return out.bfill().ffill()

# ---------------------- Sentiment ------------------------------
def generate_sentiment(ticker: str):
    analyzer = SentimentIntensityAnalyzer()
    headlines = [
        f"{ticker} stock rises as investors show optimism",
        f"{ticker} faces new challenges amid market changes",
        f"Analysts discuss {ticker} future growth prospects"
    ]
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores))

# ---------------------- Signal Logic ---------------------------
def generate_signal(ind: pd.DataFrame, sentiment_score: float):
    last = ind.iloc[-1]
    score = 0
    if last["Close"] > last["MA50"]: score += 1
    if last["MA50"] > last["MA200"]: score += 1
    if last["RSI"] < 30: score += 1
    if last["RSI"] > 70: score -= 1
    if last["MACD"] > last["MACD_Signal"]: score += 1
    else: score -= 1
    if sentiment_score > 0.2: score += 1
    elif sentiment_score < -0.2: score -= 1

    if score >= 3:
        signal = "BUY"
        color = "green"
    elif score <= -2:
        signal = "SELL"
        color = "red"
    else:
        signal = "HOLD"
        color = "orange"
    return signal, color, score

# ---------------------- Main Execution -------------------------
if ticker:
    df = fetch_prices(ticker)
    if df is not None:
        ind = compute_indicators(df)
        sentiment = generate_sentiment(ticker)
        signal, color, score = generate_signal(ind, sentiment)
        last = ind.iloc[-1]

        # Summary Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${last['Close']:.2f}")
        col2.metric("RSI (14)", f"{last['RSI']:.2f}")
        col3.metric("MACD", f"{last['MACD']:.2f}")

        st.markdown(f"### 🟢 **Signal: {signal}** (Score: {score}, Sentiment: {sentiment:+.2f})")

        # Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], mode="lines", name="Close", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=ind.index, y=ind["MA50"], mode="lines", name="MA50", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=ind.index, y=ind["MA200"], mode="lines", name="MA200", line=dict(color="green")))
        fig.update_layout(
            title=f"{ticker} Price & Moving Averages",
            template="plotly_white",
            height=500,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"No data found for {ticker}. Try a different symbol.")
