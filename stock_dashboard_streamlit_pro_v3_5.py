# stock_dashboard_streamlit_pro_v3_5.py
# ---------------------------------------------------------------
# AI Stock Signals — PRO v3.5 (Investor Edition)
# 5y data • Short vs Long-Term recommendations • Education hub
# News-aware scoring • Adaptive RSI (ATR) • Market (SPY) context
# ---------------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from typing import Dict, List, Tuple

st.set_page_config(page_title="AI Stock Signals — PRO v3.5", layout="wide")
st.title("📈 AI Stock Signals — PRO v3.5 (Investor Edition)")
st.caption("Short vs Long-Term recommendations • Live finance news sentiment • Explainable signals")

# ----------------------- UI CONTROLS ---------------------------
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Investment Horizon", ["Short-Term (1–2 weeks)", "Long-Term (3–12 months)"],
                       index=0, horizontal=True)
with c3:
    mode = st.radio("Risk Profile", ["Aggressive", "Moderate", "Conservative"],
                    index=1, horizontal=True)

use_news = st.toggle("Use Live News Sentiment", value=True)

# If you set NEWS_API_KEY in Streamlit Cloud → Settings → Secrets
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", None)

# ----------------------- FETCH PRICES --------------------------
@st.cache_data(ttl=3600)
def fetch_prices(t: str, period="5y", interval="1d") -> pd.DataFrame | None:
    try:
        df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)
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

# ----------------------- INDICATORS ----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # MAs
    out["MA10"]  = c.rolling(10,  min_periods=1).mean()
    out["MA20"]  = c.rolling(20,  min_periods=1).mean()
    out["MA50"]  = c.rolling(50,  min_periods=1).mean()
    out["MA200"] = c.rolling(200, min_periods=1).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()

    # RSI (Wilder)
    d = c.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
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

    # Bollinger (20,2)
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std(ddof=0)
    out["BB_Mid"] = bb_mid
    out["BB_Up"]  = bb_mid + 2 * bb_std
    out["BB_Low"] = bb_mid - 2 * bb_std

    # ATR
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=1).mean()

    # ADX (14) — safe
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(np.ravel(plus_dm), index=df.index)
    minus_dm = pd.Series(np.ravel(minus_dm), index=df.index)
    atr_smooth = tr.rolling(14, min_periods=1).mean()
    plus_di  = 100 * (plus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["ADX"] = dx.rolling(14, min_periods=1).mean()

    # Volume spike
    vol_ma20 = v.rolling(20, min_periods=1).mean()
    out["Vol_Spike"] = (v > 2 * vol_ma20).astype(int)

    out["Close"] = c
    return out.bfill().ffill()

# ----------------------- NEWS & EVENTS -------------------------
FINANCE_DOMAINS = [
    "finance.yahoo.com","cnbc.com","bloomberg.com","marketwatch.com",
    "reuters.com","seekingalpha.com","investing.com","wsj.com","ft.com"
]
EVENT_POS = ["upgrade","raises target","initiates buy","strong buy",
             "beats estimates","price target raised","outperform","overweight"]
EVENT_NEG = ["downgrade","misses estimates","underperform","underweight",
             "guidance cut","profit warning"]

def newsapi_url(ticker: str, api_key: str) -> str:
    q = f"{ticker} AND (stock OR shares OR earnings OR guidance OR revenue OR profit OR forecast)"
    domains = ",".join(FINANCE_DOMAINS)
    return (
        f"https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(q)}&language=en&sortBy=publishedAt&pageSize=20"
        f"&domains={domains}&apiKey={api_key}"
    )

def weighted_sentiment(articles: List[Dict]) -> Tuple[float, List[Tuple[str,str]]]:
    analyzer = SentimentIntensityAnalyzer()
    scores, top = [], []
    for a in articles[:20]:
        title = a.get("title") or ""
        desc  = a.get("description") or ""
        src   = (a.get("source") or {}).get("name", "")
        url   = a.get("url", "")
        text  = f"{title}. {desc}"
        w = 1.5 if any(k in (src or "").lower() for k in ["bloomberg","cnbc","reuters","wsj","ft.com"]) else 1.0
        s = analyzer.polarity_scores(text)["compound"]
        scores.append(w * s)
        if len(top) < 5:
            top.append((title, f"{src} | {url}"))
    return (float(np.mean(scores)) if scores else 0.0), top

def detect_events(headlines: List[str]) -> Dict[str,bool]:
    blob = " ".join(h.lower() for h in headlines)
    return {"upgrade": any(k in blob for k in EVENT_POS),
            "downgrade": any(k in blob for k in EVENT_NEG)}

@st.cache_data(ttl=900)
def fetch_news(ticker: str, api_key: str) -> Tuple[float, Dict[str,bool], List[Tuple[str,str]]]:
    try:
        r = requests.get(newsapi_url(ticker, api_key), timeout=10)
        data = r.json() if r.ok else {}
        articles = data.get("articles", []) if isinstance(data, dict) else []
        sentiment, top = weighted_sentiment(articles)
        events = detect_events([a.get("title","") for a in articles])
        return sentiment, events, top
    except Exception:
        return 0.0, {"upgrade": False, "downgrade": False}, []

# ----------------------- SIGNAL ENGINE (v3.5) ------------------
def generate_signal_v35(ind: pd.DataFrame, sentiment: float, mode: str, horizon: str,
                        events: Dict[str,bool], market_bias: float):
    last = ind.iloc[-1]
    score = 0.0

    # Risk sensitivity
    sens = {"Aggressive": 0.7, "Moderate": 1.0, "Conservative": 1.5}[mode]

    # Adaptive RSI via ATR (percent of price)
    vol_pct = float((ind["ATR"].iloc[-1] / max(1e-9, last["Close"])) * 100)
    # Short-term tighter bands, Long-term wider bands
    if "Short" in horizon:
        rsi_high = np.clip(68 + vol_pct*0.6, 65, 78)
        rsi_low  = np.clip(32 - vol_pct*0.6, 22, 35)
    else:
        rsi_high = np.clip(70 + vol_pct*0.4, 68, 80)
        rsi_low  = np.clip(30 - vol_pct*0.4, 20, 32)

    # --- Trend (heavier for Long-Term)
    if "Long" in horizon:
        if last["MA50"] > last["MA200"]: score += 1.5
        else: score -= 1.0
        if last["MA20"] > last["MA50"]: score += 0.5
    else:
        if last["MA20"] > last["MA50"]: score += 1.0
        if last["MA50"] > last["MA200"]: score += 0.75

    if last["ADX"] > (22 if "Long" in horizon else 18): score += 0.5

    # --- Momentum (heavier for Short-Term)
    if "Short" in horizon:
        if last["RSI"] < rsi_low:  score += 1.5
        if last["RSI"] > rsi_high: score -= 1.25
        score += 1 if last["MACD"] > last["MACD_Signal"] else -1
        if last["MACD_Slope"] > 0: score += 0.5
    else:
        # Long-term: reward MACD above signal, downweight RSI extremes
        score += 1 if last["MACD"] > last["MACD_Signal"] else -0.5
        if last["RSI"] < rsi_low:  score += 0.5
        if last["RSI"] > rsi_high: score -= 0.5

    # --- Mean reversion (mostly for Short-Term timing)
    if "Short" in horizon:
        if last["Close"] < last["BB_Low"]: score += 0.75
        if last["Close"] > last["BB_Up"]:  score -= 0.75

    # --- Volume confirmation
    if last["Vol_Spike"]: score += 0.5

    # --- News sentiment
    score += float(np.clip(sentiment * (2.0 if "Short" in horizon else 1.5), -1.5, 1.5))

    # --- Event boost
    if events.get("upgrade"):   score += (2.0 if "Short" in horizon else 1.5)
    if events.get("downgrade"): score -= (2.0 if "Short" in horizon else 1.5)

    # --- Market bias (SPY)
    score += 0.3 * market_bias

    # Thresholds by mode & horizon
    if "Short" in horizon:
        buy_th  = 2.0 * sens * (0.95 if market_bias > 0 else 1.0)
        sell_th = -1.0 * sens * (0.95 if market_bias < 0 else 1.0)
    else:
        buy_th  = 2.5 * sens * (0.95 if market_bias > 0 else 1.0)
        sell_th = -1.5 * sens * (0.95 if market_bias < 0 else 1.0)

    if score >= buy_th:
        return "BUY", "green", round(score, 2), (rsi_low, rsi_high)
    elif score <= sell_th:
        return "SELL", "red", round(score, 2), (rsi_low, rsi_high)
    else:
        return "HOLD", "orange", round(score, 2), (rsi_low, rsi_high)

# ----------------------- PLOTTING ------------------------------
def plot_dashboard(ind: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                        subplot_titles=("Price / MAs / Bollinger", "MACD", "RSI"))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA50"],  name="MA50",  line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA200"], name="MA200", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Up"],  name="BB Upper", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Low"], name="BB Lower", line=dict(color="gray", dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"],        name="MACD",   line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_Hist"], name="Hist", marker_color="gray", opacity=0.45), row=2, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="teal")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",  row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green",row=3, col=1)

    fig.update_layout(height=820, title=f"{ticker} — Technical Dashboard (5y)",
                      template="plotly_white", legend=dict(orientation="h", y=-0.08))
    return fig

# ----------------------- EXPLANATION ---------------------------
def explain_signal(ind: pd.DataFrame, sentiment: float, signal: str, events: Dict[str,bool],
                   market_bias: float, rsi_band: Tuple[float,float], horizon: str) -> str:
    last = ind.iloc[-1]
    low, high = rsi_band
    bullets = []

    # Horizon-aware phrasing
    if "Short" in horizon:
        bullets.append("Short-term view: momentum & timing focus")
    else:
        bullets.append("Long-term view: trend & durability focus")

    # Trend
    bullets.append("MA50 > MA200 → bullish trend" if last["MA50"] > last["MA200"]
                   else "MA50 < MA200 → bearish bias")

    # Momentum
    bullets.append("MACD rising" if last["MACD_Slope"] > 0 else "MACD flattening")
    bullets.append("MACD above signal → momentum up" if last["MACD"] > last["MACD_Signal"]
                   else "MACD below signal → momentum soft")

    # RSI band
    if last["RSI"] < low:
        bullets.append(f"RSI < {low:.0f} → oversold (potential bounce)")
    elif last["RSI"] > high:
        bullets.append(f"RSI > {high:.0f} → overbought risk")

    # Bands
    if "Short" in horizon:
        if last["Close"] < ind["BB_Low"].iloc[-1]:
            bullets.append("Below lower Bollinger → mean-reversion setup")
        elif last["Close"] > ind["BB_Up"].iloc[-1]:
            bullets.append("Above upper Bollinger → stretched")

    # News
    if events.get("upgrade"): bullets.append("Analyst upgrade / price-target raise detected")
    if events.get("downgrade"): bullets.append("Analyst downgrade / warning detected")
    if sentiment > 0.15: bullets.append(f"Finance sentiment positive ({sentiment:+.2f})")
    elif sentiment < -0.15: bullets.append(f"Finance sentiment negative ({sentiment:+.2f})")

    # Market
    bullets.append("Market uptrend (SPY) supports longs" if market_bias > 0
                   else ("Market downtrend (SPY) adds caution" if market_bias < 0 else "Market neutral"))

    md = "**Why {}:**\n".format(signal)
    for b in bullets:
        md += f"- {b}\n"
    return md

def confidence_from_score(score: float, signal: str):
    conf = float(min(abs(score) / 3.5, 1.0))
    if signal == "BUY":
        text = "Breakout/upgrade scenario — partial entry ok" if conf > 0.6 else "Cautious buy — wait for pullback"
    elif signal == "SELL":
        text = "Bearish confluence" if conf > 0.6 else "Weakness developing — confirm breakdown"
    else:
        text = "Indecision — wait for better risk/reward"
    return conf, text

# ----------------------- BACKTEST PREVIEW ----------------------
def backtest_preview(ind: pd.DataFrame) -> float:
    test = ind.copy()
    rsi = test["RSI"]
    rsi_high_med = float(np.nanmedian(rsi.tail(100))) + 10
    rsi_low_med  = float(np.nanmedian(rsi.tail(100))) - 10
    test["Sig"] = 0
    test.loc[rsi < rsi_low_med, "Sig"] = 1
    test.loc[rsi > rsi_high_med, "Sig"] = -1
    test["Next"] = test["Close"].shift(-5)
    test["Ret"]  = (test["Next"] - test["Close"]) / test["Close"]
    mask = test["Sig"] != 0
    if mask.sum() < 10:
        return 0.0
    acc = (np.sign(test.loc[mask,"Ret"]) == test.loc[mask,"Sig"]).mean()
    return float(round(acc * 100, 1))

# ========================= MAIN VIEW ===========================
tab_dash, tab_learn = st.tabs(["📊 Dashboard", "🎓 Learn Indicators"])

with tab_dash:
    if ticker:
        df = fetch_prices(ticker)
        if df is None:
            st.error(f"No data found for {ticker}. Try another symbol.")
        else:
            ind = compute_indicators(df)

            # Market context
            spy = fetch_index("SPY")
            market_bias = 0.0
            if spy is not None and len(spy) > 200:
                spy_ind = compute_indicators(spy)
                if spy_ind.iloc[-1]["MA50"] > spy_ind.iloc[-1]["MA200"]:
                    market_bias = 1.0
                elif spy_ind.iloc[-1]["MA50"] < spy_ind.iloc[-1]["MA200"]:
                    market_bias = -1.0

            # News
            sentiment, events, headlines = 0.0, {"upgrade": False, "downgrade": False}, []
            news_note = ""
            if use_news:
                if NEWS_API_KEY:
                    sentiment, events, headlines = fetch_news(ticker, NEWS_API_KEY)
                else:
                    news_note = "🔒 Add NEWS_API_KEY in Streamlit Secrets to enable live finance headlines."

            # Signal
            signal, color, score, rsi_band = generate_signal_v35(ind, sentiment, mode, horizon, events, market_bias)
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
                f"### **{signal}**  "
                f"(Score: {score}, Sentiment: {sentiment:+.2f}, Market: "
                f"{'Bull' if market_bias>0 else ('Bear' if market_bias<0 else 'Neutral')}, "
                f"Horizon: *{horizon}*)"
            )

            fig = plot_dashboard(ind, ticker)
            st.plotly_chart(fig, use_container_width=True)

            acc = backtest_preview(ind)
            st.write(f"**Backtest Preview (adaptive RSI swing, 5-day):** {acc:.1f}% accuracy")

            st.markdown(explain_signal(ind, sentiment, signal, events, market_bias, rsi_band, horizon))
            conf, conf_text = confidence_from_score(score, signal)
            st.progress(conf)
            st.write(f"**AI Confidence:** {conf*100:.0f}% — {conf_text}")

            if use_news:
                st.markdown("#### Latest Finance Headlines")
                if headlines:
                    for title, src in headlines:
                        st.write(f"- {title}  \n  <span style='color:#888;font-size:0.9em'>{src}</span>", unsafe_allow_html=True)
                else:
                    st.info(news_note or "No recent finance headlines found.")

with tab_learn:
    st.subheader("How to Read the Signals (Cheat Sheet)")
    with st.expander("RSI — Relative Strength Index"):
        st.markdown("""
**What it measures:** Momentum (speed of recent gains vs losses).  
**Key levels:**  
- **> 70**: Overbought (risk of pullback)  
- **< 30**: Oversold (rebound likely)  
**Tip:** In strong trends, RSI can stay overbought/oversold. Adaptive bands help avoid false signals.
        """)
    with st.expander("MACD — Moving Average Convergence Divergence"):
        st.markdown("""
**What it measures:** Trend momentum via short vs long EMAs.  
- **MACD > Signal**: Bullish crossover (momentum up)  
- **MACD < Signal**: Bearish crossover (momentum down)  
- **Histogram**: Strength of the divergence  
**Tip:** Combine with RSI for high-probability entries.
        """)
    with st.expander("Moving Averages (MA20/50/200)"):
        st.markdown("""
**What they measure:** Directional trend over time windows.  
- **MA50 > MA200**: Long-term uptrend (bullish)  
- **Price > MA50**: Near-term strength  
**Tip:** Crossovers + MACD confirmation improve reliability.
        """)
    with st.expander("Bollinger Bands"):
        st.markdown("""
**What they measure:** Volatility envelope around the mid (MA20).  
- **Price < Lower Band**: Oversold → mean-reversion setup  
- **Price > Upper Band**: Overbought → risk of cooling off  
**Tip:** Great for **short-term timing** in range-bound markets.
        """)
    with st.expander("ADX — Trend Strength"):
        st.markdown("""
**What it measures:** Strength of trend (not direction).  
- **> 20–25**: Trend is meaningful  
**Tip:** Use ADX to decide whether to favor **breakouts** (high ADX) or **mean-reversion** (low ADX).
        """)
    with st.expander("ATR — Volatility Range"):
        st.markdown("""
**What it measures:** Typical daily move size.  
**Use cases:** Position sizing, stop distances, adaptive RSI bands.
        """)
    with st.expander("Volume & Sentiment"):
        st.markdown("""
**Volume spikes** confirm strong moves.  
**Sentiment** from finance headlines helps incorporate fundamentals & events (earnings, upgrades, guidance).
        """)
    st.info("Pro tip: Choose **Short-Term** for swing entries/exits, **Long-Term** for core holdings & DCA decisions.")

# ----------------------- DISCLAIMER ----------------------------
st.markdown("---")
st.markdown("""
**Disclaimer:** This app is for **educational purposes only** and is **not financial advice**.  
Always do your own research or consult a licensed financial advisor.
© 2025 Raj Gupta — *AI Stock Signals — PRO v3.5*
""")
