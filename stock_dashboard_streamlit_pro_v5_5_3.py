import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests, feedparser
from io import StringIO

# ============= Page config =============
st.set_page_config(page_title="AI Stock Signals — PRO v5.5.3", layout="wide")
st.title("🧠📊 AI Stock Signals — PRO v5.5.3")
st.caption("Technicals • Macro • News • Analyst • Hybrid AI Forecast • Adaptive DCA")

# ============= Inputs =============
c1, c2, c3 = st.columns([2,2,3])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Simulation amount ($)", min_value=500, max_value=50_000, step=500, value=10_000)

# ============= Chart Timeframe Selector =============
timeframes = {
    "1D": ("1d", "1m"),
    "1W": ("7d", "5m"),
    "1M": ("1mo", "30m"),
    "3M": ("3mo", "1h"),
    "6M": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y": ("1y", "1d"),
    "2Y": ("2y", "1d"),
    "5Y": ("5y", "1d"),
    "10Y": ("10y", "1d"),
    "ALL": ("max", "1d"),
}
tf = st.selectbox("Chart Timeframe", list(timeframes.keys()), index=5)
period, interval = timeframes[tf]

# ============= Data fetchers =============
@st.cache_data(ttl=7200)
def fetch_prices_tf(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return None
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    try: df.index = df.index.tz_localize(None)
    except Exception: pass
    return df

@st.cache_data(ttl=86400)
def fetch_earnings_date(ticker: str):
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if "Earnings Date" in cal.index:
            return str(cal.loc["Earnings Date"].values[0])
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_major_indices():
    indices = {
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC"
    }
    data = {}
    for name, symbol in indices.items():
        try:
            df = yf.download(symbol, period="1d", interval="1m", progress=False)
            last = df.iloc[-1]
            data[name] = {
                "Open": last["Open"],
                "High": last["High"],
                "Low": last["Low"],
                "Close": last["Close"],
                "Volume": last["Volume"]
            }
        except Exception:
            data[name] = None
    return data

@st.cache_data(ttl=86400)
def fetch_fundamentals(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "Open": info.get("open"),
            "High": info.get("dayHigh"),
            "Low": info.get("dayLow"),
            "Volume": info.get("volume"),
            "P/E": info.get("trailingPE"),
            "Market Cap": info.get("marketCap"),
            "52w High": info.get("fiftyTwoWeekHigh"),
            "52w Low": info.get("fiftyTwoWeekLow"),
            "Avg Vol": info.get("averageVolume"),
            "Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "EPS": info.get("trailingEps"),
        }
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def fetch_macro():
    macro = {}
    try:
        vix = yf.download("^VIX", period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
        macro["vix_last"] = float(vix.iloc[-1]) if not vix.empty else None
    except Exception:
        macro["vix_last"] = None
    try:
        spx = yf.download("^GSPC", period="6mo", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
        if len(spx) >= 20:
            ma5 = float(spx.rolling(5).mean().iloc[-1])
            ma20 = float(spx.rolling(20).mean().iloc[-1])
            macro["spx_trend"] = "Bullish" if ma5 > ma20 else "Bearish"
            macro["spx_5d_vs_20d"] = round(((ma5 - ma20) / ma20) * 100, 2)
        else:
            macro["spx_trend"], macro["spx_5d_vs_20d"] = None, None
    except Exception:
        macro["spx_trend"], macro["spx_5d_vs_20d"] = None, None
    def fred_csv_last(series_id: str):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.rename(columns={series_id: "value"}).dropna()
        df = df[df["value"] != "."]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna()
    try:
        cpi_df = fred_csv_last("CPIAUCSL").sort_values("DATE")
        last = cpi_df.iloc[-1]["value"]
        prev12 = cpi_df.iloc[-13]["value"] if len(cpi_df) > 13 else np.nan
        macro["cpi_yoy"] = round(((last/prev12) - 1) * 100, 2) if prev12 == prev12 else None
        macro["cpi_date"] = cpi_df.iloc[-1]["DATE"].date().isoformat()
    except Exception:
        macro["cpi_yoy"], macro["cpi_date"] = None, None
    try:
        un_df = fred_csv_last("UNRATE").sort_values("DATE")
        macro["unemp_rate"] = round(float(un_df.iloc[-1]["value"]), 2)
        macro["unemp_date"] = un_df.iloc[-1]["DATE"].date().isoformat()
    except Exception:
        macro["unemp_rate"], macro["unemp_date"] = None, None
    if macro["cpi_yoy"] is None: macro["cpi_yoy"] = 3.2
    if macro["unemp_rate"] is None: macro["unemp_rate"] = 3.8
    return macro

# ============= Indicators =============
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out["MA20"]  = c.rolling(20, min_periods=1).mean()
    out["MA50"]  = c.rolling(50, min_periods=1).mean()
    out["MA200"] = c.rolling(200, min_periods=1).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    out["RSI"] = (100 - (100/(1+rs))).fillna(50)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"]   = out["MACD"] - out["MACD_Signal"]
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std(ddof=0)
    out["BB_Up"]  = bb_mid + 2*bb_std
    out["BB_Low"] = bb_mid - 2*bb_std
    prev_close = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=1).mean()
    up_move   = h.diff()
    down_move = -l.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(np.ravel(plus_dm).astype(float),  index=df.index)
    minus_dm = pd.Series(np.ravel(minus_dm).astype(float), index=df.index)
    atr_smooth = tr.rolling(14, min_periods=1).mean()
    plus_di  = 100 * (plus_dm.rolling(14, min_periods=1).sum()  / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["ADX"] = dx.rolling(14, min_periods=1).mean()
    try:
        width = (out["BB_Up"] - out["BB_Low"]) / c.replace(0, np.nan)
        if isinstance(width, pd.DataFrame):
            width = width.iloc[:, 0]
        out["BB_Width"] = pd.Series(width, index=df.index).astype(float).fillna(0)
    except Exception:
        out["BB_Width"] = pd.Series(0.0, index=df.index)
    out["Close"] = c
    return out.bfill().ffill()

# ============= News + Sentiment =============
def fetch_news_and_sentiment(ticker: str):
    analyzer = SentimentIntensityAnalyzer()
    headlines, scores = [], []
    api_key = st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None
    if api_key:
        try:
            url = (
                "https://newsapi.org/v2/everything?"
                f"q={ticker}&language=en&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
            )
            r = requests.get(url, timeout=10)
            if r.ok:
                for a in r.json().get("articles", [])[:10]:
                    title = a.get("title") or ""
                    url_  = a.get("url") or ""
                    src   = a.get("source", {}).get("name", "News")
                    pub   = a.get("publishedAt", "")
                    headlines.append({"title": title, "url": url_, "source": src, "published": pub})
                    scores.append(analyzer.polarity_scores(title)["compound"])
        except Exception:
            pass
    if not headlines:
        feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]
        for feed in feeds:
            try:
                d = feedparser.parse(feed)
                for e in d.entries[:6]:
                    title = e.title
                    link  = getattr(e, "link", "")
                    src   = "Yahoo Finance" if "yahoo" in feed else "CNBC"
                    pub   = getattr(e, "published", "")
                    headlines.append({"title": title, "url": link, "source": src, "published": pub})
                    scores.append(analyzer.polarity_scores(title)["compound"])
            except Exception:
                continue
    sentiment = float(np.mean(scores)) if scores else 0.0
    return headlines[:10], sentiment

# ============= Analyst pulse (best-effort) =============
@st.cache_data(ttl=86400)
def analyst_pulse(ticker: str):
    try:
        t = yf.Ticker(ticker)
        rec = getattr(t, "recommendations", None)
        if rec is None or rec.empty: return {"buy_ratio": None, "samples": 0}
        df = rec.tail(200).copy()
        df.columns = [c.lower() for c in df.columns]
        actions = df.get("action")
        if actions is None: return {"buy_ratio": None, "samples": 0}
        actions = actions.astype(str).str.lower()
        ups = actions.str.contains("upgrade").sum()
        downs = actions.str.contains("downgrade").sum()
        tot = ups + downs
        return {"buy_ratio": (ups/tot) if tot>0 else None, "samples": int(tot)}
    except Exception:
        return {"buy_ratio": None, "samples": 0}

def market_confidence(sentiment: float, buy_ratio: float | None):
    sent_norm = (sentiment + 1) / 2
    conf = 0.6 * sent_norm + 0.4 * (buy_ratio if buy_ratio is not None else 0.5)
    return max(0,min(1,conf))

# ============= Signal logic =============
def generate_signal(ind: pd.DataFrame, sentiment: float, horizon: str):
    last = ind.iloc[-1]
    score = 0.0
    if last["MA20"] > last["MA50"]:  score += 1
    if last["MA50"] > last["MA200"]: score += 1
    if last["ADX"] > 25:             score += 1
    if last["RSI"] < 30: score += 1.2
    elif last["RSI"] > 70: score -= 1.2
    if last["MACD"] > last["MACD_Signal"]: score += 1
    else: score -= 1
    if last["Close"] < last["BB_Low"]: score += 0.8
    elif last["Close"] > last["BB_Up"]: score -= 0.8
    if last["BB_Width"] > 0.12: score -= 0.2
    if ind["Vol_Spike"].iloc[-1] if "Vol_Spike" in ind else 0: score += 0.3
    score += float(np.clip(sentiment, -0.8, 0.8))
    th_buy, th_sell = (2.5, -2.0) if "Short" in horizon else (3.5, -2.5)
    if score >= th_buy:  return "BUY", "green", round(score, 2)
    if score <= th_sell: return "SELL", "red",  round(score, 2)
    return "HOLD", "orange", round(score, 2)

def confidence_from_score(score: float) -> float:
    return float(min(abs(score) / 5.0, 1.0))

def explain_signal_verbose(ind, sentiment, decision, horizon):
    last = ind.iloc[-1]
    reasons = []
    if last["MA50"] > last["MA200"]:
        reasons.append("✅ **Uptrend** — MA50 above MA200 (long-term strength).")
    else:
        reasons.append("⚠️ **Downtrend** — MA50 below MA200 (bearish bias).")
    if last["MACD"] > last["MACD_Signal"]:
        reasons.append("✅ **MACD bullish crossover** — momentum improving.")
    else:
        reasons.append("⚠️ **MACD bearish** — momentum fading.")
    if last["RSI"] < 30:
        reasons.append("✅ **RSI oversold** (<30) — potential rebound zone.")
    elif last["RSI"] > 70:
        reasons.append("⚠️ **RSI overbought** (>70) — may need cooldown.")
    elif 45 <= last["RSI"] <= 55:
        reasons.append("💤 **RSI neutral** — sideways momentum.")
    bb_width = last.get("BB_Width", 0)
    if bb_width < 0.05:
        reasons.append("🔹 **Bollinger squeeze** — volatility contraction, breakout possible.")
    elif last["Close"] < last["BB_Low"]:
        reasons.append("✅ **Price below lower band** — mean reversion likely.")
    elif last["Close"] > last["BB_Up"]:
        reasons.append("⚠️ **Price above upper band** — extended move, possible pullback.")
    if last["ADX"] > 25:
        reasons.append("✅ **Strong trend** (ADX>25)
