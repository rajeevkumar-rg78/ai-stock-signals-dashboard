# stock_dashboard_streamlit_pro_v2.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------------
st.set_page_config(page_title="AI Stock Signals PRO v2", layout="wide")
st.title("📈 AI Stock Signals Dashboard — PRO v2")
st.caption("Enhanced BUY/SELL logic, 2-year trend analysis, ADX fix, and backtest preview.")

# ---------------------------------------------------------------
# User Input
# ---------------------------------------------------------------
c1, c2 = st.columns([3, 2])
with c1:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL").upper()
with c2:
    mode = st.radio("Trading Mode", ["Conservative", "Moderate", "Aggressive"], index=1, horizontal=True)

# ---------------------------------------------------------------
# Data Fetch
# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_prices(ticker: str, period="2y", interval="1d") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ---------------------------------------------------------------
# Indicator Calculations
# ---------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # Moving Averages
    out["MA20"] = c.rolling(20, min_periods=1).mean()
    out["MA50"] = c.rolling(50, min_periods=1).mean()
    out["MA200"] = c.rolling(200, min_periods=1).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # Bollinger Bands
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std(ddof=0)
    out["BB_Mid"] = bb_mid
    out["BB_Up"] = bb_mid + 2 * bb_std
    out["BB_Low"] = bb_mid - 2 * bb_std

    # ATR
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=1).mean()

    # --- ADX (safe flatten fix) ---
    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # flatten ensures 1D array (avoids ValueError)
    plus_dm = np.ravel(plus_dm)
    minus_dm = np.ravel(minus_dm)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr_smooth = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["ADX"] = dx.rolling(14, min_periods=1).mean()

    # Volume Spike
    vol_ma20 = v.rolling(20, min_periods=1).mean()
    out["Vol_Spike"] = (v > 2 * vol_ma20).astype(int)

    out["Close"] = c
    return out.bfill().ffill()

# ---------------------------------------------------------------
# Sentiment (static demo)
# ---------------------------------------------------------------
def generate_sentiment(ticker: str) -> float:
    analyzer = SentimentIntensityAnalyzer()
    headlines = [
        f"{ticker} stock rises as investors remain optimistic",
        f"Analysts warn about valuation pressure for {ticker}",
        f"{ticker} continues to attract institutional buying"
    ]
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores))

# ---------------------------------------------------------------
# Signal Logic
# ---------------------------------------------------------------
def generate_signal(ind: pd.DataFrame, sentiment_score: float, mode: str):
    last = ind.iloc[-1]
    score = 0.0

    sensitivity = {"Aggressive": 0.7, "Moderate": 1.0, "Conservative": 1.5}[mode]

    # Trend
    if last["MA20"] > last["MA50"]: score += 1
    if last["MA50"] > last["MA200"]: score += 1
    if last["ADX"] > 20: score += 0.5

    # Momentum
    if last["RSI"] < 35: score += 1.5
    elif last["RSI"] > 65: score -= 1.5

    # MACD
    if last["MACD"] > last["MACD_Signal"]: score += 1
    else: score -= 1

    # Mean Reversion
    if last["Close"] < last["BB_Low"]: score += 1
    elif last["Close"] > last["BB_Up"]: score -= 1

    # Volume Spike
    if last["Vol_Spike"]: score += 0.5

    # Sentiment
    score += float(np.clip(sentiment_score, -0.5, 0.5))

    # Thresholds adapt by mode
    buy_th = 2.5 * sensitivity
    sell_th = -1.5 * sensitivity

    if score >= buy_th:
        return "BUY", "green", round(score, 2)
    elif score <= sell_th:
        return "SELL", "red", round(score, 2)
    else:
        return "HOLD", "orange", round(score, 2)

# ---------------------------------------------------------------
# Backtest Preview
# ---------------------------------------------------------------
def backtest_preview(ind: pd.DataFrame) -> float:
    test = ind.copy()
    test["Sig"] = 0
    test.loc[test["RSI"] < 35, "Sig"] = 1
    test.loc[test["RSI"] > 65, "Sig"] = -1
    test["Next"] = test["Close"].shift(-5)
    test["Ret"] = (test["Next"] - test["Close"]) / test["Close"]
    mask = test["Sig"] != 0
    if mask.sum() < 5:
        return 0.0
    acc = (np.sign(test.loc[mask, "Ret"]) == test.loc[mask, "Sig"]).mean()
    return float(round(acc * 100, 1))

# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------
def plot_indicators(ind: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05,
                        subplot_titles=("Price / MAs / Bollinger", "MACD", "RSI"))

    # Price section
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA50"], name="MA50", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA200"], name="MA200", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Up"], name="BB Upper", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Low"], name="BB Lower", line=dict(color="gray", dash="dot")), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"], name="MACD", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_Hist"], name="Hist", marker_color="gray", opacity=0.45), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="teal")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=820, title=f"{ticker} — Technical Dashboard (2y view)",
                      template="plotly_white", legend=dict(orientation="h", y=-0.08))
    return fig

# ---------------------------------------------------------------
# Explanation & Confidence
# ---------------------------------------------------------------
def explain_signal(ind: pd.DataFrame, sentiment: float, name: str) -> str:
    last = ind.iloc[-1]
    reasons = []
    reasons.append("MA50 > MA200 → bullish trend" if last["MA50"] > last["MA200"] else "MA50 < MA200 → bearish bias")
    reasons.append("MACD above signal → upward momentum" if last["MACD"] > last["MACD_Signal"] else "MACD below signal → momentum soft")
    if last["RSI"] < 35:
        reasons.append("RSI low → oversold zone")
    elif last["RSI"] > 65:
        reasons.append("RSI high → overbought risk")
    if last["Close"] < last["BB_Low"]:
        reasons.append("Price under lower Bollinger → possible rebound")
    elif last["Close"] > last["BB_Up"]:
        reasons.append("Price above upper Bollinger → stretched")
    reasons.append("Sentiment positive" if sentiment > 0.1 else ("Sentiment negative" if sentiment < -0.1 else "Neutral sentiment"))
    return f"**Why {name}:** " + ", ".join(reasons)

def confidence_from_score(score: float, signal: str):
    conf = float(min(abs(score) / 3.5, 1.0))
    if signal == "BUY":
        text = "High-conviction entry zone" if conf > 0.7 else "Cautious buy — confirm trend"
    elif signal == "SELL":
        text = "Strong bearish setup" if conf > 0.7 else "Early warning — confirm breakdown"
    else:
        text = "Market indecision — wait for confirmation"
    return conf, text

# ---------------------------------------------------------------
# Main App
# ---------------------------------------------------------------
if ticker:
    df = fetch_prices(ticker)
    if df is None:
        st.error(f"No data found for {ticker}. Try another symbol.")
    else:
        ind = compute_indicators(df)
        sentiment = generate_sentiment(ticker)
        signal, color, score = generate_signal(ind, sentiment, mode)
        last = ind.iloc[-1]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Price", f"${last['Close']:.2f}")
        col2.metric("RSI (14)", f"{last['RSI']:.2f}")
        col3.metric("MACD", f"{last['MACD']:.2f}")
        col4.metric("ADX", f"{last['ADX']:.1f}")
        col5.metric("ATR (14)", f"{last['ATR']:.2f}")

        st.markdown(f"### 🟢 **Signal: {signal}** (Score: {score}, Sentiment: {sentiment:+.2f}, Mode: {mode})")

        fig = plot_indicators(ind, ticker)
        st.plotly_chart(fig, use_container_width=True)

        accuracy = backtest_preview(ind)
        st.write(f"**Backtest Preview (RSI swing 5-day):** {accuracy:.1f}% accuracy")

        st.markdown(explain_signal(ind, sentiment, signal))
        conf, conf_text = confidence_from_score(score, signal)
        st.progress(conf)
        st.write(f"**AI Confidence:** {conf*100:.0f}% — {conf_text}")

# ---------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
**Disclaimer:**  
This dashboard is for **educational and informational purposes only**.  
It does **not constitute financial advice**.  
Always do your own research or consult a licensed financial advisor before making investment decisions.

© 2025 Raj Gupta — *AI Stock Signals Dashboard PRO v2.1*
"""
)
