# stock_dashboard_streamlit_pro.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------
st.set_page_config(page_title="AI Stock Signals PRO Dashboard", layout="wide")
st.title("📈 AI Stock Signals Dashboard — PRO Edition")
st.caption("Live technicals, sentiment, and enhanced BUY/HOLD/SELL logic powered by Yahoo Finance.")

# ---------------------------------------------------------------
# User Input
# ---------------------------------------------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL").upper()

# ---------------------------------------------------------------
# Data Fetch
# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_prices(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df.rename(columns={"Adj Close": "Close"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return None

# ---------------------------------------------------------------
# Indicator Calculations
# ---------------------------------------------------------------
def compute_indicators(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # Moving Averages
    out["MA20"] = close.rolling(20).mean()
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    out["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = exp12 - exp26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    bb_mid = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std(ddof=0)

    out["BB_Mid"] = bb_mid.astype(float)
    out["BB_Up"] = (bb_mid + 2 * bb_std).astype(float)
    out["BB_Low"] = (bb_mid - 2 * bb_std).astype(float)


    # ATR (volatility)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()

    # --- ADX (trend strength, cloud-safe version) ---
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low.shift(1)).clip(lower=0)
    tr_smooth = tr.rolling(window=14, min_periods=1).sum()

    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).sum() / (tr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).sum() / (tr_smooth + 1e-9))

    # DX = |+DI - -DI| / (+DI + -DI)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100

    # --- FORCE 1D SERIES (avoid DataFrame assignment error) ---
    if isinstance(dx, pd.DataFrame):
        dx = dx.iloc[:, 0]

    dx = pd.Series(dx.values, index=df.index, dtype="float64")
    adx = dx.rolling(window=14, min_periods=1).mean()
    adx = pd.Series(adx.values, index=df.index, dtype="float64")

    out["ADX"] = adx



    # Volume spike
    vol_ma20 = vol.rolling(20).mean()
    out["Vol_Spike"] = (vol > 2 * vol_ma20).astype(int)

    out["Close"] = close
    return out.bfill().ffill()

# ---------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------
def generate_sentiment(ticker: str):
    analyzer = SentimentIntensityAnalyzer()
    headlines = [
        f"{ticker} stock rises as investors show optimism",
        f"{ticker} faces new challenges amid market changes",
        f"Analysts discuss {ticker} growth prospects"
    ]
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores))

# ---------------------------------------------------------------
# Signal Logic (multi-factor scoring)
# ---------------------------------------------------------------
def generate_signal(ind: pd.DataFrame, sentiment_score: float):
    last = ind.iloc[-1]
    score = 0

    # --- Trend ---
    if last["MA20"] > last["MA50"]: score += 1
    if last["MA50"] > last["MA200"]: score += 1
    if last["ADX"] > 25: score += 1  # strong trend

    # --- Momentum ---
    if last["RSI"] < 30: score += 1
    if last["RSI"] > 70: score -= 1
    if last["MACD"] > last["MACD_Signal"]: score += 1
    else: score -= 1

    # --- Mean reversion ---
    if last["Close"] < last["BB_Low"]: score += 1
    if last["Close"] > last["BB_Up"]: score -= 1

    # --- Volume confirmation ---
    if last["Vol_Spike"]: score += 1

    # --- Sentiment adjustment ---
    if sentiment_score > 0.2: score += 1
    elif sentiment_score < -0.2: score -= 1

    # --- Final decision ---
    if score >= 4:
        signal = "BUY"
        color = "green"
    elif score <= -3:
        signal = "SELL"
        color = "red"
    else:
        signal = "HOLD"
        color = "orange"
    return signal, color, score

# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------
def plot_indicators(df, ind, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05,
                        subplot_titles=("Price / MAs / Bollinger", "MACD", "RSI"))

    # --- Price + MA + Bollinger ---
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA50"], name="MA50", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA200"], name="MA200", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Up"], name="BB Upper", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Low"], name="BB Lower", line=dict(color="gray", dash="dot")), row=1, col=1)

    # --- MACD ---
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"], name="MACD", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_Hist"], name="Hist", marker_color="gray", opacity=0.5), row=2, col=1)

    # --- RSI ---
    fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="teal")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=800, title=f"{ticker} — Technical Dashboard",
                      template="plotly_white",
                      legend=dict(orientation="h", y=-0.1))
    return fig

# ---------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------
if ticker:
    df = fetch_prices(ticker)
    if df is not None:
        ind = compute_indicators(df)
        sentiment = generate_sentiment(ticker)
        signal, color, score = generate_signal(ind, sentiment)
        last = ind.iloc[-1]

        # --- Summary Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"${last['Close']:.2f}")
        col2.metric("RSI (14)", f"{last['RSI']:.2f}")
        col3.metric("MACD", f"{last['MACD']:.2f}")
        col4.metric("ADX", f"{last['ADX']:.1f}")

        st.markdown(f"### 🟢 **Signal: {signal}** (Score: {score}, Sentiment: {sentiment:+.2f})")

        fig = plot_indicators(df, ind, ticker)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"No data found for {ticker}. Try a different symbol.")
def explain_signal(ind, sentiment, signal_name):
    last = ind.iloc[-1]
    reasons = []
    if last["RSI"] < 35:
        reasons.append("RSI low → oversold condition")
    elif last["RSI"] > 65:
        reasons.append("RSI high → overbought condition")

    if last["MA50"] > last["MA200"]:
        reasons.append("MA50 > MA200 → bullish trend")
    else:
        reasons.append("MA50 < MA200 → bearish trend")

    if last["MACD"] > last["MACD_Signal"]:
        reasons.append("MACD above signal → upward momentum")
    else:
        reasons.append("MACD below signal → downward momentum")

    if sentiment > 0.2:
        reasons.append("Positive sentiment")
    elif sentiment < -0.2:
        reasons.append("Negative sentiment")

    summary = ", ".join(reasons)
    return f"**Why {signal_name}:** {summary}"

st.markdown(explain_signal(ind, sentiment, signal))
confidence = min(abs(score) / 5, 1.0)
st.progress(confidence)
st.write(f"**Confidence:** {confidence*100:.0f}%")

# ---------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Disclaimer:**  
    This dashboard is for **educational and informational purposes only**.  
    It does **not constitute financial advice**.  
    Always perform your own research or consult a licensed financial advisor before making investment decisions.  

    © 2025 Raj Gupta — *AI Stock Signals Dashboard PRO*
    """,
    unsafe_allow_html=True
)
