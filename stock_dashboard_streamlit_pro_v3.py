# stock_dashboard_streamlit_pro_v3.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from typing import Dict, List, Tuple

# ---------------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------------
st.set_page_config(page_title="AI Stock Signals PRO v3", layout="wide")
st.title("📈 AI Stock Signals — PRO v3")
st.caption("5y context • Adaptive RSI • News/Upgrade boost • Finance-weighted sentiment • Market-aware thresholds")

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    mode = st.radio("Trading Mode", ["Conservative", "Moderate", "Aggressive"], index=1, horizontal=True)
with c3:
    use_news = st.toggle("Use Live News Sentiment", value=True)

# Secrets (set in Streamlit Cloud → Settings → Secrets)
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", None)

# ---------------------------------------------------------------
# Data Fetch
# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_prices(ticker: str, period="5y", interval="1d") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_index(symbol="SPY", period="5y", interval="1d") -> pd.DataFrame | None:
    return fetch_prices(symbol, period=period, interval=interval)

# ---------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # Moving averages
    out["MA20"] = c.rolling(20, min_periods=1).mean()
    out["MA50"] = c.rolling(50, min_periods=1).mean()
    out["MA200"] = c.rolling(200, min_periods=1).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()

    # RSI (Wilder)
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
    out["MACD_Slope"] = out["MACD"].diff()

    # Bollinger (20, 2)
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std(ddof=0)
    out["BB_Mid"] = bb_mid
    out["BB_Up"] = bb_mid + 2 * bb_std
    out["BB_Low"] = bb_mid - 2 * bb_std

    # ATR (14)
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=1).mean()

    # ADX (14) — safe flatten
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(np.ravel(plus_dm), index=df.index)
    minus_dm = pd.Series(np.ravel(minus_dm), index=df.index)
    atr_smooth = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["ADX"] = dx.rolling(14, min_periods=1).mean()

    # Volume spike
    vol_ma20 = v.rolling(20, min_periods=1).mean()
    out["Vol_Spike"] = (v > 2 * vol_ma20).astype(int)

    out["Close"] = c
    return out.bfill().ffill()

# ---------------------------------------------------------------
# News sentiment (finance-weighted) + event detection
# ---------------------------------------------------------------
FINANCE_DOMAINS = [
    "finance.yahoo.com", "cnbc.com", "bloomberg.com", "marketwatch.com",
    "reuters.com", "seekingalpha.com", "investing.com", "wsj.com", "ft.com"
]
EVENT_KEYWORDS_POS = [
    "upgrade", "raises target", "initiates buy", "strong buy",
    "beats estimates", "price target raised", "outperform", "overweight"
]
EVENT_KEYWORDS_NEG = [
    "downgrade", "misses estimates", "underperform", "underweight",
    "guidance cut", "profit warning"
]

def build_newsapi_url(ticker: str, api_key: str) -> str:
    q = f"{ticker} AND (stock OR shares OR earnings OR guidance OR revenue OR profit OR forecast)"
    domains = ",".join(FINANCE_DOMAINS)
    return (
        f"https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(q)}"
        f"&language=en&sortBy=publishedAt"
        f"&pageSize=20"
        f"&domains={domains}"
        f"&apiKey={api_key}"
    )

def compute_weighted_sentiment(articles: List[Dict]) -> Tuple[float, List[str], List[Tuple[str,str]]]:
    analyzer = SentimentIntensityAnalyzer()
    weighted_scores = []
    headlines = []
    top5 = []

    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        src = (a.get("source") or {}).get("name", "") + " | " + (a.get("url", "") or "")
        text = f"{title}. {desc}"

        # weight: top finance outlets heavier
        source_lower = src.lower()
        w = 1.5 if any(dom in source_lower for dom in ["bloomberg", "cnbc", "reuters", "wsj", "ft.com"]) else 1.0

        s = analyzer.polarity_scores(text)["compound"]
        weighted_scores.append(w * s)
        headlines.append(title)
        if len(top5) < 5:
            top5.append((title, src))

    sentiment = float(np.mean(weighted_scores)) if weighted_scores else 0.0
    return sentiment, headlines, top5

def detect_events(headlines: List[str]) -> Dict[str, bool]:
    h_blob = " ".join(h.lower() for h in headlines)
    has_pos = any(k in h_blob for k in EVENT_KEYWORDS_POS)
    has_neg = any(k in h_blob for k in EVENT_KEYWORDS_NEG)
    return {"upgrade": has_pos, "downgrade": has_neg}

@st.cache_data(ttl=900)
def fetch_news_and_sentiment(ticker: str, api_key: str) -> Tuple[float, Dict[str,bool], List[Tuple[str,str]]]:
    try:
        url = build_newsapi_url(ticker, api_key)
        r = requests.get(url, timeout=10)
        data = r.json()
        articles = data.get("articles", []) if isinstance(data, dict) else []
        sentiment, headlines, top5 = compute_weighted_sentiment(articles)
        events = detect_events(headlines)
        return sentiment, events, top5
    except Exception:
        return 0.0, {"upgrade": False, "downgrade": False}, []

# ---------------------------------------------------------------
# Signal Logic (adaptive RSI, event boost, market-aware)
# ---------------------------------------------------------------
def generate_signal(ind: pd.DataFrame, sentiment: float, mode: str,
                    events: Dict[str,bool], market_bias: float):
    last = ind.iloc[-1]
    score = 0.0

    # Mode sensitivity (lower = more trades)
    sens = {"Aggressive": 0.7, "Moderate": 1.0, "Conservative": 1.5}[mode]

    # Adaptive RSI thresholds based on current volatility (ATR/Price)
    vol_ratio = float((ind["ATR"].iloc[-1] / max(1e-9, last["Close"])) * 100)  # %
    rsi_high = np.clip(70 + vol_ratio, 68, 80)  # cap expansion
    rsi_low  = np.clip(30 - vol_ratio, 20, 32)

    # Trend
    if last["MA20"] > last["MA50"]: score += 1
    if last["MA50"] > last["MA200"]: score += 1
    if last["ADX"] > 20: score += 0.5

    # Momentum
    if last["RSI"] < rsi_low: score += 1.5
    elif last["RSI"] > rsi_high: score -= 1.25

    # MACD + slope
    score += 1 if last["MACD"] > last["MACD_Signal"] else -1
    if last["MACD_Slope"] > 0: score += 0.5

    # Mean reversion
    if last["Close"] < last["BB_Low"]: score += 0.75
    elif last["Close"] > last["BB_Up"]: score -= 0.75

    # Volume
    if last["Vol_Spike"]: score += 0.5

    # News sentiment (finance-weighted)
    score += float(np.clip(sentiment * 2, -1.5, 1.5))

    # Event boosts (e.g., upgrade/target raise)
    if events.get("upgrade", False): score += 2.0
    if events.get("downgrade", False): score -= 2.0

    # Market bias (SPY trend) gently adjusts score and thresholds
    score += 0.3 * market_bias  # market uptrend adds a nudge

    # Thresholds adapt by mode and market (slightly looser in uptrends)
    buy_th = 2.3 * sens * (0.95 if market_bias > 0 else 1.0)
    sell_th = -1.4 * sens * (0.95 if market_bias < 0 else 1.0)

    if score >= buy_th:
        return "BUY", "green", round(score, 2), (rsi_low, rsi_high)
    elif score <= sell_th:
        return "SELL", "red", round(score, 2), (rsi_low, rsi_high)
    else:
        return "HOLD", "orange", round(score, 2), (rsi_low, rsi_high)

# ---------------------------------------------------------------
# Backtest Preview (unchanged, simple)
# ---------------------------------------------------------------
def backtest_preview(ind: pd.DataFrame) -> float:
    test = ind.copy()
    test["Sig"] = 0
    # use adaptive-ish RSI bands median over last 100 bars for robustness
    rsi = test["RSI"]
    rsi_high_med = float(np.nanmedian(rsi.tail(100))) + 10
    rsi_low_med = float(np.nanmedian(rsi.tail(100))) - 10
    test.loc[rsi < rsi_low_med, "Sig"] = 1
    test.loc[rsi > rsi_high_med, "Sig"] = -1
    test["Next"] = test["Close"].shift(-5)
    test["Ret"] = (test["Next"] - test["Close"]) / test["Close"]
    mask = test["Sig"] != 0
    if mask.sum() < 10:
        return 0.0
    acc = (np.sign(test.loc[mask, "Ret"]) == test.loc[mask, "Sig"]).mean()
    return float(round(acc * 100, 1))

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
def plot_indicators(ind: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05,
                        subplot_titles=("Price / MAs / Bollinger", "MACD", "RSI"))

    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA50"], name="MA50", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA200"], name="MA200", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Up"], name="BB Upper", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Low"], name="BB Lower", line=dict(color="gray", dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"], name="MACD", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_Hist"], name="Hist", marker_color="gray", opacity=0.45), row=2, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="teal")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=820, title=f"{ticker} — Technical Dashboard (5y)",
                      template="plotly_white", legend=dict(orientation="h", y=-0.08))
    return fig

# ---------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------
def explain_signal(ind: pd.DataFrame, sentiment: float, name: str, events: Dict[str,bool],
                   market_bias: float, rsi_band: Tuple[float,float]) -> str:
    last = ind.iloc[-1]
    reasons = []
    reasons.append("MA50 > MA200 → bullish trend" if last["MA50"] > last["MA200"] else "MA50 < MA200 → bearish bias")
    reasons.append("MACD rising" if last["MACD_Slope"] > 0 else "MACD flattening")
    reasons.append("MACD above signal → momentum up" if last["MACD"] > last["MACD_Signal"] else "MACD below signal → momentum soft")
    low, high = rsi_band
    if last["RSI"] < low:
        reasons.append(f"RSI < {low:.0f} → oversold")
    elif last["RSI"] > high:
        reasons.append(f"RSI > {high:.0f} → overbought risk")
    if last["Close"] < last["BB_Low"]:
        reasons.append("Below lower Bollinger → rebound risk-on")
    elif last["Close"] > last["BB_Up"]:
        reasons.append("Above upper Bollinger → stretched")
    if events.get("upgrade"):
        reasons.append("Analyst upgrade/event boost detected")
    if events.get("downgrade"):
        reasons.append("Downgrade / negative event detected")
    if sentiment > 0.15:
        reasons.append(f"Positive finance sentiment ({sentiment:+.2f})")
    elif sentiment < -0.15:
        reasons.append(f"Negative finance sentiment ({sentiment:+.2f})")
    reasons.append("Market uptrend (SPY) supports longs" if market_bias > 0 else "Market downtrend (SPY) adds caution")
    return f"**Why {name}:** " + ", ".join(reasons)

def confidence_from_score(score: float, signal: str):
    conf = float(min(abs(score) / 3.5, 1.0))
    if signal == "BUY":
        text = "Breakout/upgrade scenario — partial entry ok" if conf > 0.6 else "Cautious buy — wait for pullback"
    elif signal == "SELL":
        text = "Strong bearish confluence" if conf > 0.6 else "Weakness developing — confirm breakdown"
    else:
        text = "Indecision — wait for better risk/reward"
    return conf, text

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if ticker:
    df = fetch_prices(ticker)
    if df is None:
        st.error(f"No data found for {ticker}. Try another symbol.")
    else:
        ind = compute_indicators(df)

        # Market context (SPY)
        spy = fetch_index("SPY")
        market_bias = 0.0
        if spy is not None and len(spy) > 200:
            spy_ind = compute_indicators(spy)
            spy_last = spy_ind.iloc[-1]
            if spy_last["MA50"] > spy_last["MA200"]:
                market_bias = 1.0
            elif spy_last["MA50"] < spy_last["MA200"]:
                market_bias = -1.0

        # News sentiment & events
        sentiment, events, headlines = 0.0, {"upgrade": False, "downgrade": False}, []
        news_note = ""
        if use_news:
            if NEWS_API_KEY:
                sentiment, events, headlines = fetch_news_and_sentiment(ticker, NEWS_API_KEY)
            else:
                news_note = "🔒 Add NEWS_API_KEY to Streamlit Secrets to enable live news."

        # Signal
        signal, color, score, rsi_band = generate_signal(ind, sentiment, mode, events, market_bias)
        last = ind.iloc[-1]

        # Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Price", f"${last['Close']:.2f}")
        m2.metric("RSI (14)", f"{last['RSI']:.1f}")
        m3.metric("MACD", f"{last['MACD']:.2f}")
        m4.metric("ADX", f"{last['ADX']:.1f}")
        m5.metric("ATR (14)", f"{last['ATR']:.2f}")
        m6.metric("Mode", mode)

        st.markdown(
            f"### **Signal: {signal}**  "
            f"(Score: {score}, Sentiment: {sentiment:+.2f}, Market bias: {'Bull' if market_bias>0 else ('Bear' if market_bias<0 else 'Neutral')})"
        )

        # Chart
        fig = plot_indicators(ind, ticker)
        st.plotly_chart(fig, use_container_width=True)

        # Backtest preview
        acc = backtest_preview(ind)
        st.write(f"**Backtest Preview (adaptive RSI swing, 5-day):** {acc:.1f}% accuracy")

        # Explanation + confidence
        st.markdown(explain_signal(ind, sentiment, signal, events, market_bias, rsi_band))
        conf, conf_text = confidence_from_score(score, signal)
        st.progress(conf)
        st.write(f"**AI Confidence:** {conf*100:.0f}% — {conf_text}")

        # Headlines
        if use_news:
            st.markdown("#### Latest Finance Headlines")
            if headlines:
                for title, src in headlines:
                    st.write(f"- {title}  \n  <span style='color:#888;font-size:0.9em'>{src}</span>", unsafe_allow_html=True)
            else:
                st.info(news_note or "No recent finance headlines found.")

# ---------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
**Disclaimer:**  
This tool is for **educational purposes only** and is **not financial advice**.
Always do your own research or consult a licensed financial advisor.

© 2025 Raj Gupta — *AI Stock Signals PRO v3*
"""
)
