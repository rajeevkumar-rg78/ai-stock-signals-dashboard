# ============================================================
# stock_dashboard_streamlit_pro_v5_5_final.py
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests, feedparser
from io import StringIO
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# Streamlit Config
# ============================================================
st.set_page_config(page_title="AI Stock Signals — PRO v5.5 Final", layout="wide")
st.title("📈 AI Stock Signals — PRO v5.5 Final")
st.caption("Technicals • Macro • News Sentiment • Analyst Pulse • AI Forecast • Adaptive DCA")

# ============================================================
# User Input
# ============================================================
c1, c2, c3 = st.columns([2,2,3])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Simulation amount ($)", min_value=500, max_value=50_000, step=500, value=10_000)

# ============================================================
# Data Fetch
# ============================================================
@st.cache_data(ttl=7200)
def fetch_prices(ticker, horizon):
    period = "6mo" if "Short" in horizon else "5y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return None
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df

# ============================================================
# Indicators
# ============================================================
def compute_indicators(df):
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    out["MA20"]  = c.rolling(20).mean()
    out["MA50"]  = c.rolling(50).mean()
    out["MA200"] = c.rolling(200).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out["RSI"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std(ddof=0)
    out["BB_Up"] = bb_mid + 2 * bb_std
    out["BB_Low"] = bb_mid - 2 * bb_std
    # ✅ Fix for 1-D alignment error
    bb_width = ((out["BB_Up"] - out["BB_Low"]) / c.replace(0, np.nan)).fillna(0)
    bb_width = np.ravel(bb_width)  # <-- flatten to 1D
    out["BB_Width"] = pd.Series(bb_width, index=df.index)

    prev_close = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()

    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    atr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).sum() / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).sum() / (atr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["ADX"] = dx.rolling(14).mean()

    vol_ma = v.rolling(20).mean()
    out["Vol_Spike"] = (v > 2 * vol_ma).astype(int)
    out["Close"] = c
    return out.bfill().ffill()

# ============================================================
# News Sentiment
# ============================================================
def fetch_news_and_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    headlines, scores = [], []
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    ]
    for feed in feeds:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:5]:
                title = e.title
                url = getattr(e, "link", "")
                src = "Yahoo Finance" if "yahoo" in feed else "CNBC"
                headlines.append({"title": title, "url": url, "src": src})
                scores.append(analyzer.polarity_scores(title)["compound"])
        except Exception:
            continue
    sentiment = float(np.mean(scores)) if scores else 0.0
    return headlines[:10], sentiment

# ============================================================
# Signal Logic
# ============================================================
def generate_signal(ind, sentiment, horizon):
    last = ind.iloc[-1]
    score = 0
    if last["MA20"] > last["MA50"]: score += 1
    if last["MA50"] > last["MA200"]: score += 1
    if last["ADX"] > 25: score += 1
    if last["RSI"] < 30: score += 1.2
    elif last["RSI"] > 70: score -= 1.2
    if last["MACD"] > last["MACD_Signal"]: score += 1
    else: score -= 1
    if last["Close"] < last["BB_Low"]: score += 0.8
    elif last["Close"] > last["BB_Up"]: score -= 0.8
    if last["Vol_Spike"]: score += 0.3
    score += float(np.clip(sentiment, -0.8, 0.8))
    if "Short" in horizon:
        th_buy, th_sell = 2.5, -2.0
    else:
        th_buy, th_sell = 3.5, -2.5
    if score >= th_buy: return "BUY", "green", round(score, 2)
    if score <= th_sell: return "SELL", "red", round(score, 2)
    return "HOLD", "orange", round(score, 2)

# ============================================================
# AI Forecast (safe version)
# ============================================================
def ai_forecast(df, ind):
    try:
        df, ind = df.align(ind, join="inner", axis=0)
        X = np.column_stack([
            ind["RSI"], ind["MACD"], ind["MACD_Signal"], ind["MA20"], ind["MA50"], ind["ADX"]
        ])
        y = df["Close"].shift(-1).fillna(method="ffill")
        if len(X) < 200 or np.isnan(X).any():
            return {"range": None, "conf": 0.0}
        model = RandomForestRegressor(n_estimators=80, random_state=42)
        model.fit(X[:-5], y[:-5])
        preds = model.predict(X[-5:])
        if np.isnan(preds).all():
            return {"range": None, "conf": 0.0}
        low, mid, high = np.nanpercentile(preds, [25, 50, 75])
        conf = round(float(model.score(X[:-5], y[:-5]) * 100), 1)
        return {"range": (low, mid, high), "conf": conf}
    except Exception as e:
        print("AI forecast error:", e)
        return {"range": None, "conf": 0.0}

# ============================================================
# Main Execution
# ============================================================
if not ticker:
    st.stop()

df = fetch_prices(ticker, horizon)
if df is None:
    st.error("No data found. Try another symbol.")
    st.stop()

ind = compute_indicators(df)
headlines, sentiment = fetch_news_and_sentiment(ticker)
decision, color, score = generate_signal(ind, sentiment, horizon)
ai = ai_forecast(df, ind)

# ============================================================
# Dashboard Display
# ============================================================
last = ind.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"${last['Close']:.2f}")
col2.metric("RSI", f"{last['RSI']:.1f}")
col3.metric("ADX", f"{last['ADX']:.1f}")
col4.metric("MACD", f"{last['MACD']:.2f}")

st.markdown(f"### **Signal: {decision}** (Score {score:+.2f}, Sentiment {sentiment:+.2f})")

if ai["range"] and not any(np.isnan(ai["range"])):
    low, mid, high = ai["range"]
    st.success(f"**Expected 5-day range:** ${low:.2f} – ${mid:.2f} – ${high:.2f}")
else:
    st.info("Not enough data for 5-day AI range forecast.")

st.write(f"**AI Confidence:** {ai['conf']:.1f}%")

# ============================================================
# News Display
# ============================================================
with st.expander("📰 Latest News Headlines"):
    if not headlines:
        st.write("No headlines found.")
    else:
        for h in headlines:
            st.markdown(f"- [{h['title']}]({h['url']}) — *{h['src']}*")

# ============================================================
# Learn
# ============================================================
with st.expander("📘 Learn: Indicators & Strategy"):
    st.markdown("""
**RSI** – <30 oversold, >70 overbought  
**MACD** – momentum/trend crossovers  
**Bollinger Bands** – volatility extremes  
**ADX** – trend strength indicator  
**ATR** – volatility for dynamic targets  
**AI Forecast** – short-term model-based price projection  
**Adaptive DCA** – more investment at deep oversold zones  
""")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("© 2025 Raj Gupta — AI Stock Signals PRO v5.5 Final • Educational use only • Not financial advice")
