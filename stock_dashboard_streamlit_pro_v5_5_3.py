import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests, feedparser
from io import StringIO
import plotly.graph_objects as go

def render_analyst_pulse(pulse: dict):
    """Ultra-compact Analyst Pulse — single-line display."""
    if not pulse or pulse.get("samples", 0) <= 0:
        st.info("No analyst data available.")
        return

    buy = pulse.get("buy") or 0
    hold = pulse.get("hold") or 0
    sell = pulse.get("sell") or 0
    total = max(buy + hold + sell, 1e-9)
    buy_pct, hold_pct, sell_pct = [round(x / total * 100, 1) for x in (buy, hold, sell)]

    st.markdown(f"""
    <div style='background-color:#fff;border:1px solid #e5e5e5;border-radius:8px;
                padding:6px 10px;margin-top:4px;box-shadow:0 1px 2px rgba(0,0,0,0.04);'>
        <div style='display:flex;align-items:center;gap:10px;'>
            <div style='font-size:13px;color:#666;white-space:nowrap;'>
                <b>Analyst Pulse</b> • {pulse['samples']} ratings
            </div>
            <div style='flex:1;height:10px;border-radius:5px;overflow:hidden;display:flex;'>
                <div style='width:{buy_pct}%;background-color:#28a745;'></div>
                <div style='width:{hold_pct}%;background-color:#f0ad4e;'></div>
                <div style='width:{sell_pct}%;background-color:#dc3545;'></div>
            </div>
            <div style='font-size:12.5px;color:#333;text-align:right;white-space:nowrap;'>
                🟢 {buy_pct}% | ⚪ {hold_pct}% | 🔴 {sell_pct}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def human_fmt(val, kind=None):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if kind == "vol":
            # Volume: show as M or K
            if val >= 1e9:
                return f"{val/1e9:.2f}B"
            elif val >= 1e6:
                return f"{val/1e6:.2f}M"
            elif val >= 1e3:
                return f"{val/1e3:.2f}K"
            else:
                return f"{val:.0f}"
        if kind == "cap":
            # Market Cap: show as T, B, M
            if val >= 1e12:
                return f"{val/1e12:.3f}T"
            elif val >= 1e9:
                return f"{val/1e9:.2f}B"
            elif val >= 1e6:
                return f"{val/1e6:.2f}M"
            else:
                return f"{val:.0f}"
        if kind == "pct":
            # Percent
            return f"{val*100:.2f}%"
        return f"{val:.2f}"
    except Exception:
        return "—"


# ============= Utility functions =============
def safe_fmt(val, fmt="{:.2f}", default="—"):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return fmt.format(val)
    except Exception:
        return default



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
        # If calendar is a dict and has "Earnings Date"
        if isinstance(cal, dict) and "Earnings Date" in cal:
            val = cal["Earnings Date"]
            # If it's a list, get the first element
            if isinstance(val, (list, np.ndarray)):
                val = val[0]
            # If it's a datetime.date, format it
            if hasattr(val, "strftime"):
                return val.strftime("%Y-%m-%d")
            return str(val)
        return None
    except Exception as e:
        st.write("Earnings date error:", e)
        return None


@st.cache_data(ttl=3600)
def fetch_major_indices():
    indices = {
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "S&P 500": "^GSPC",
        "VXN (Nasdaq Volatility)": "^VXN"
    }
    data = {}
    for name, symbol in indices.items():
        try:
            df = yf.download(symbol, period="1mo", interval="1d", progress=False)
            if df.empty:
                data[name] = None
                continue
            last_row = df.iloc[-1]
            data[name] = {
                "Open": float(last_row["Open"]),
                "High": float(last_row["High"]),
                "Low": float(last_row["Low"]),
                "Close": float(last_row["Close"]),
                "Volume": float(last_row["Volume"])
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
    """Enhanced analyst consensus detector with multiple fallbacks — fixed for DataFrame truth issue."""
    try:
        t = yf.Ticker(ticker)

        # 1️⃣ Try historical recommendations (rarely populated)
        rec = getattr(t, "recommendations", None)
        if rec is not None and not rec.empty:
            df = rec.tail(200).copy()
            df.columns = [c.lower() for c in df.columns]
            col = None
            for candidate in ["to grade", "action"]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col:
                grades = df[col].astype(str).str.lower()
                total = len(grades)
                buy_terms = ["buy", "strong buy", "outperform", "overweight", "add", "accumulate", "long-term buy", "top pick"]
                hold_terms = ["hold", "neutral", "market perform", "equal weight", "sector perform", "peer perform"]
                sell_terms = ["sell", "underperform", "underweight", "reduce", "weak hold", "short", "negative"]
                buy = grades.str.contains("|".join(buy_terms)).sum()
                hold = grades.str.contains("|".join(hold_terms)).sum()
                sell = grades.str.contains("|".join(sell_terms)).sum()
                return {
                    "buy": buy / total if total > 0 else None,
                    "hold": hold / total if total > 0 else None,
                    "sell": sell / total if total > 0 else None,
                    "neutral": hold / total if total > 0 else None,
                    "samples": total
                }

        # 2️⃣ Try the modern recommendations_summary
        trend = getattr(t, "recommendations_summary", None)
        if trend is not None:
            # Handle both dict and DataFrame forms
            if isinstance(trend, pd.DataFrame):
                if "strongBuy" in trend.columns:
                    row = trend.iloc[0].to_dict()
                else:
                    row = trend.to_dict(orient="records")[0]
            elif isinstance(trend, dict):
                row = trend
            else:
                row = {}

            if row:
                total = sum(v for v in row.values() if isinstance(v, (int, float)))
                if total > 0:
                    buy = (row.get("buy", 0) + row.get("strongBuy", 0)) / total
                    hold = row.get("hold", 0) / total
                    sell = (row.get("sell", 0) + row.get("strongSell", 0)) / total
                    return {
                        "buy": buy, "hold": hold, "sell": sell,
                        "neutral": hold, "samples": total
                    }

        # 3️⃣ Fallback: direct Yahoo API JSON
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=recommendationTrend"
        r = requests.get(url, timeout=10)
        if r.ok:
            js = r.json()
            trend_list = js.get("quoteSummary", {}).get("result", [{}])[0] \
                            .get("recommendationTrend", {}).get("trend", [])
            if trend_list:
                latest = trend_list[-1]
                total = sum(latest.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
                if total > 0:
                    buy = (latest.get("buy", 0) + latest.get("strongBuy", 0)) / total
                    hold = latest.get("hold", 0) / total
                    sell = (latest.get("sell", 0) + latest.get("strongSell", 0)) / total
                    return {
                        "buy": buy, "hold": hold, "sell": sell,
                        "neutral": hold, "samples": total
                    }

        # 4️⃣ Default: no valid data
        return {"buy": None, "hold": None, "sell": None, "neutral": None, "samples": 0}

    except Exception as e:
        st.write("Analyst pulse error:", e)
        return {"buy": None, "hold": None, "sell": None, "neutral": None, "samples": 0}


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

    # --- Trend structure ---
    if last["MA50"] > last["MA200"]:
        reasons.append("✅ **Uptrend** — MA50 above MA200 (long-term strength).")
    else:
        reasons.append("⚠️ **Downtrend** — MA50 below MA200 (bearish bias).")

    # --- MACD ---
    if last["MACD"] > last["MACD_Signal"]:
        reasons.append("✅ **MACD bullish crossover** — momentum improving.")
    else:
        reasons.append("⚠️ **MACD bearish** — momentum fading.")

    # --- RSI ---
    if last["RSI"] < 30:
        reasons.append("✅ **RSI oversold** (<30) — potential rebound zone.")
    elif last["RSI"] > 70:
        reasons.append("⚠️ **RSI overbought** (>70) — may need cooldown.")
    elif 45 <= last["RSI"] <= 55:
        reasons.append("💤 **RSI neutral** — sideways momentum.")

    # --- Bollinger analysis ---
    bb_width = last.get("BB_Width", 0)
    if bb_width < 0.05:
        reasons.append("🔹 **Bollinger squeeze** — volatility contraction, breakout possible.")
    elif last["Close"] < last["BB_Low"]:
        reasons.append("✅ **Price below lower band** — mean reversion likely.")
    elif last["Close"] > last["BB_Up"]:
        reasons.append("⚠️ **Price above upper band** — extended move, possible pullback.")

    # --- ADX (trend strength) ---
    if last["ADX"] > 25:
        reasons.append("✅ **Strong trend** (ADX>25) — price movement has conviction.")
    else:
        reasons.append("💤 **Weak trend** (ADX<25) — possible range-bound action.")

    # --- Cup & Handle / Double Bottom heuristic ---
    c = ind["Close"].tail(50)
    if len(c) > 20:
        lows = c.rolling(5).min()
        if lows.iloc[-1] > lows.min() and lows.idxmin() < lows.index[-10]:
            reasons.append("📈 **Possible Double Bottom** pattern forming (support retest).")
        rolling_mean = c.rolling(20).mean()
        if c.iloc[-1] > rolling_mean.iloc[-1] and (c.iloc[-1] - rolling_mean.iloc[-1]) / rolling_mean.iloc[-1] < 0.05:
            reasons.append("☕ **Cup & Handle-like** recovery — consolidation breakout zone.")

    # --- News & sentiment ---
    if sentiment > 0.1:
        reasons.append(f"📰 **Positive sentiment** ({sentiment:+.2f}) — news tone supportive.")
    elif sentiment < -0.1:
        reasons.append(f"⚠️ **Negative sentiment** ({sentiment:+.2f}) — cautious outlook.")
    else:
        reasons.append("📄 **Neutral news sentiment** — limited bias from headlines.")

    # --- Horizon context ---
    reasons.append("🎯 Strategy tuned for **short-term swing** moves (3–10d)." if "Short" in horizon
                   else "🏦 Strategy tuned for **long-term accumulation** (>3mo).")

    return "\n".join(reasons)

# ============= AI Forecast (robust, no NaN) =============
def ai_forecast(df: pd.DataFrame, ind: pd.DataFrame):
    """Tiny, robust forecaster: bootstrap last 120d daily returns to estimate 5d range."""
    r = df["Close"].pct_change().dropna()

    # --- Guarantee 1-D numeric array ---
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    r = pd.to_numeric(r, errors="coerce").dropna()

    if len(r) < 30:
        return {"pred_move": 0.0, "conf": 0.0, "range": None}

    r_hist = r.tail(120).values.flatten()  # <— ensures 1D array
    sims = np.random.choice(r_hist, size=(1000, 5), replace=True).sum(axis=1)
    mu = float(np.mean(sims))
    sd = float(np.std(sims))
    low = mu - 1.96 * sd
    high = mu + 1.96 * sd
    conf = float(min(1.0, abs(mu) / (sd + 1e-9)))
    return {"pred_move": mu, "conf": conf, "range": (low, mu, high)}


# ============= Chart =============
def plot_dashboard(ind: pd.DataFrame, ticker: str, zones=True):
    last = ind.iloc[-1]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                        subplot_titles=("Price / MAs / Bollinger + Zones", "MACD", "RSI"))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color="blue")), 1,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA50"],  name="MA50",  line=dict(color="orange")), 1,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA200"], name="MA200", line=dict(color="green")), 1,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Up"],  name="BB Upper", line=dict(color="gray", dash="dot")), 1,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Low"], name="BB Lower", line=dict(color="gray", dash="dot")), 1,1)
    if zones:
        buy_zone   = last["Close"] - 1.5*last["ATR"]
        target     = last["Close"] + 2.0*last["ATR"]
        stop_loss  = last["Close"] - 2.5*last["ATR"]
        fig.add_hline(y=buy_zone,    line_color="dodgerblue", line_dash="dash", annotation_text="Buy Zone",   row=1, col=1)
        fig.add_hline(y=target,      line_color="seagreen",   line_dash="dash", annotation_text="Target",     row=1, col=1)
        fig.add_hline(y=stop_loss,   line_color="crimson",    line_dash="dash", annotation_text="Stop Loss",  row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"],        name="MACD",   line=dict(color="purple")), 2,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", line=dict(color="orange")), 2,1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_Hist"], name="Hist", marker_color="gray", opacity=0.45), 2,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="teal")), 3,1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.update_layout(height=820, template="plotly_white", title=f"{ticker} — Technical Dashboard")
    return fig

# ============= Main flow =============
if not ticker:
    st.stop()

df = fetch_prices_tf(ticker, period, interval)

if df is None:
    st.error("No data found. Try another symbol.")
    st.stop()

ind = compute_indicators(df)
macro = fetch_macro()
headlines, news_sent = fetch_news_and_sentiment(ticker)
decision, color, score = generate_signal(ind, news_sent, horizon)

pulse = analyst_pulse(ticker)  # <-- Call the enhanced function here

conf_overall = market_confidence(news_sent, pulse["buy"])
ai = ai_forecast(df, ind)


# Macro header
m1, m2, m3, m4 = st.columns(4)
m1.metric("VIX (volatility)", f"{macro['vix_last']:.2f}" if macro["vix_last"] is not None else "—")
m2.metric("S&P 5d vs 20d", f"{macro['spx_5d_vs_20d']:+.2f}%" if macro["spx_5d_vs_20d"] is not None else "—",
          macro["spx_trend"] or "")
m3.metric("CPI YoY", f"{macro['cpi_yoy']:.2f}%")
m4.metric("Unemployment", f"{macro['unemp_rate']:.2f}%")

# ============= Display Earnings Date, Major Indices, Fundamentals =============

earnings_date = fetch_earnings_date(ticker)
indices = fetch_major_indices()
fund = fetch_fundamentals(ticker)

with st.expander("📅 Earnings & Indices", expanded=False):
    st.metric("Next Earnings Date", earnings_date if earnings_date else "Not available")

    idx_cols = st.columns(len(indices))
    for i, (name, data) in enumerate(indices.items()):
        if data:
            idx_cols[i].metric(f"{name} Close", human_fmt(data.get('Close')))
            idx_cols[i].metric(f"{name} High", human_fmt(data.get('High')))
            idx_cols[i].metric(f"{name} Low", human_fmt(data.get('Low')))
            idx_cols[i].metric(f"{name} Volume", human_fmt(data.get('Volume'), kind="vol"))
        else:
            idx_cols[i].metric(f"{name} Close", "Not available")
            idx_cols[i].metric(f"{name} High", "Not available")
            idx_cols[i].metric(f"{name} Low", "Not available")
            idx_cols[i].metric(f"{name} Volume", "Not available")


with st.expander("📊 Stock Fundamentals", expanded=False):
    fcols = st.columns(13)
    fcols[0].metric("Open", human_fmt(fund.get('Open')))
    fcols[1].metric("High", human_fmt(fund.get('High')))
    fcols[2].metric("Low", human_fmt(fund.get('Low')))
    fcols[3].metric("Volume", human_fmt(fund.get('Volume'), kind="vol"))
    fcols[4].metric("P/E", human_fmt(fund.get('P/E')))
    fcols[5].metric("Market Cap", human_fmt(fund.get('Market Cap'), kind="cap"))
    fcols[6].metric("52w High", human_fmt(fund.get('52w High')))
    fcols[7].metric("52w Low", human_fmt(fund.get('52w Low')))
    fcols[8].metric("Avg Vol", human_fmt(fund.get('Avg Vol'), kind="vol"))
    fcols[9].metric("Yield", human_fmt(fund.get('Yield'), kind="pct"))
    fcols[10].metric("Beta", human_fmt(fund.get('Beta')))
    fcols[11].metric("EPS", human_fmt(fund.get('EPS')))





# Snapshot metrics
last = ind.iloc[-1]
cA, cB, cC, cD, cE, cF = st.columns(6)
cA.metric("Price", f"${last['Close']:.2f}")
cB.metric("RSI (14)", f"{last['RSI']:.1f}")
cC.metric("MACD", f"{last['MACD']:.2f}")
cD.metric("ADX", f"{last['ADX']:.1f}")
cE.metric("ATR (14)", f"{last['ATR']:.2f}")
# Analyst Pulse in the last column
if pulse["samples"] > 0 and any(pulse[k] is not None for k in ["buy", "hold", "sell", "neutral"]):
    cF.metric(
        "Analyst Pulse",
        f"Buy: {int((pulse['buy'] or 0)*100)}% | Hold: {int((pulse['hold'] or 0)*100)}% | Sell: {int((pulse['sell'] or 0)*100)}% | Neutral: {int((pulse['neutral'] or 0)*100)}%"
    )
else:
    cF.metric("Analyst Pulse", "No recent analyst recommendations")

st.markdown("### 🧭 Analyst Pulse")
render_analyst_pulse(pulse)

# Signal banner with numeric target/stop
st.markdown(f"### **Signal: {decision}** (Score {score:+.2f}, News {news_sent:+.2f})")
sig_conf = confidence_from_score(score)
st.progress(conf_overall, text=f"Market Confidence {int(conf_overall*100)}% — sentiment/analyst blend")
st.metric("Signal Strength", f"{int(sig_conf*100)}%", delta=f"{score:+.2f}")

# Numeric target/stop values (ATR-based)
target_up   = last["Close"] + 2.0*last["ATR"]
buy_zone    = last["Close"] - 1.5*last["ATR"]
stop_loss   = last["Close"] - 2.5*last["ATR"]
st.write(f"📈 **Target (≈5d)**: ${target_up:.2f}  🟦 **Buy zone**: ${buy_zone:.2f}  🛑 **Stop**: ${stop_loss:.2f}")

# Chart
st.plotly_chart(plot_dashboard(ind, ticker, zones=True), use_container_width=True)

# WHY section
st.markdown("### 🧩 Why this signal")
st.markdown(explain_signal_verbose(ind, news_sent, decision, horizon))

# Forecast AI tab-like block (robust)
st.markdown("### 🤖 Forecast AI (5-day)")
st.write(f"Predicted Move (avg): {ai['pred_move']*100:+.2f}%")
if ai["range"] is not None and not any(np.isnan(ai["range"])):
    lo, mu, hi = ai["range"]
    st.write(f"Expected range in 5d: {lo*100:+.2f}% — {mu*100:+.2f}% — {hi*100:+.2f}%")
else:
    st.info("Not enough recent data for a reliable range forecast. Try a longer history or different ticker.")
st.metric("AI Confidence", f"{int(ai['conf']*100)}%")
# ============================================================
# 💵 Adaptive DCA Simulator
# ============================================================
st.markdown("## 💵 Adaptive DCA Simulator (long-only) — with partial take-profit")

def adaptive_dca_simulator(df: pd.DataFrame, ind: pd.DataFrame, cash_start: float):
    df, ind = df.align(ind, join="inner", axis=0)
    cash, shares = float(cash_start), 0.0
    trades, equity_curve = [], []
    peak_equity, halt_buys = cash_start, False

    for dt in df.index:
        price = float(df.loc[dt, "Close"])
        rsi, macd, macds = float(ind.loc[dt, "RSI"]), float(ind.loc[dt, "MACD"]), float(ind.loc[dt, "MACD_Signal"])
        ma20, ma50 = float(ind.loc[dt, "MA20"]), float(ind.loc[dt, "MA50"])
        bb_low, atr = float(ind.loc[dt, "BB_Low"]), float(ind.loc[dt, "ATR"])

        # Buy rules
        if not halt_buys:
            momentum_buy = (macd > macds and ma20 > ma50)
            oversold_buy = (rsi < 45) or (price < bb_low)
            alloc = 0.0
            if momentum_buy or oversold_buy:
                if rsi < 25: alloc = 0.30
                elif rsi < 35: alloc = 0.20
                elif rsi < 45: alloc = 0.10
            invest = cash * alloc
            if invest > 0:
                buy_shares = invest / price
                shares += buy_shares
                cash -= invest
                trades.append({"date": dt.strftime("%Y-%m-%d"), "side": "BUY", "price": round(price,2),
                               "invested": round(invest,2), "shares": round(buy_shares,6)})

        # Partial take-profit
        target_price = float(ind["Close"].iloc[-1] + 2*ind["ATR"].iloc[-1])
        if shares > 0 and price >= target_price:
            sell_shares = shares * 0.20
            proceeds = sell_shares * price
            shares -= sell_shares
            cash += proceeds
            trades.append({"date": dt.strftime("%Y-%m-%d"), "side": "SELL", "price": round(price,2),
                           "invested": -round(proceeds,2), "shares": -round(sell_shares,6)})

        equity = float(shares * price + cash)
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        dd_pct = (equity - peak_equity) / (peak_equity if peak_equity else 1)
        if dd_pct < -0.30:  # stop buying if >30% drawdown
            halt_buys = True

    final_value = shares * df["Close"].iloc[-1] + cash
    total_invested = cash_start - cash if cash_start >= cash else cash_start
    pnl = final_value - total_invested
    roi_pct = (pnl / total_invested * 100) if total_invested > 0 else 0.0
    ec = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(ec) if ec.size else np.array([0])
    dd = (ec - running_max) / np.where(running_max == 0, 1, running_max)
    max_dd = float(np.min(dd)) if dd.size else 0.0
    trades_df = pd.DataFrame(trades)
    return dict(final_value=final_value, total_invested=total_invested,
                roi_pct=roi_pct, max_drawdown_pct=round(100*max_dd,2), trades=trades_df)

# --- Run simulator
sim = adaptive_dca_simulator(df, ind, invest_amount)
# --- Defensive helper: convert safely to float ---
def safe_float(x, default=0.0):
    try:
        # Works for pandas/numpy scalars, None, or non-numeric
        return float(x)
    except Exception:
        return default

# --- Extract Adaptive DCA results safely ---
fv  = safe_float(sim.get("final_value", 0.0))
ti  = safe_float(sim.get("total_invested", 0.0))
roi = safe_float(sim.get("roi_pct", 0.0))
dd  = safe_float(sim.get("max_drawdown_pct", 0.0))

# --- Display key DCA metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Portfolio Value", f"${fv:,.2f}")
c2.metric("Total Invested", f"${ti:,.2f}")
c3.metric("ROI", f"{roi:.1f}%")
c4.metric("Max Drawdown", f"{dd:.1f}%")
if not sim["trades"].empty:
    st.dataframe(sim["trades"], use_container_width=True)
else:
    st.info("No trades executed in this period by adaptive rules.")

# Headlines
with st.expander("🗞️ Latest Headlines"):
    if not headlines:
        st.write("No headlines available.")
    else:
        for h in headlines:
            title = h["title"]; url = h["url"]; src = h.get("source",""); pub = h.get("published","")
            nice = pub[:10] if pub else ""
            st.markdown(f"- [{title}]({url}) — *{src}* {('• '+nice) if nice else ''}")
# ============================================================
# 📘 Learn (Education)
# ============================================================

def render_learn_section():
    learn_md = """
### What you’re seeing
- **Signal Tab** uses trend (MA, ADX), momentum (RSI, MACD), extremes (Bollinger), and news sentiment.  
- **Forecast AI Tab** blends historical returns + Monte Carlo (bootstrap) with a probabilistic range.  
- **Simulator Tab** models Adaptive Dollar-Cost Averaging (DCA) + partial take-profit.  

### Educational notes
- **RSI** — <30 oversold, >70 overbought.  
- **MACD** — momentum/trend crossovers.  
- **Bollinger Bands** — ±2σ around 20D mean.  
- **ADX** — trend strength (>25 = strong).  
- **ATR** — volatility; for stop/target bands.  
- **Markov chain** — probability that tomorrow continues today’s direction.  
- **Random/Monte Carlo** — random resampling of historical returns to forecast potential future range.  

### Chart Patterns
- **Cup & Handle** — rounded base followed by shallow pullback; breakout confirms bullish continuation.  
- **Double Bottom** — two similar lows with a mid-peak; breakout above the midpoint confirms reversal.  
- **Bollinger Squeeze** — narrow bands often precede strong breakouts.  
- **ADX** — measures trend strength; >25 = strong, <20 = range-bound.  
"""
    st.markdown(learn_md)

with st.expander("📘 Learn: Indicators, Patterns & AI Logic", expanded=False):
    render_learn_section()
# ============================================================
# 🧾 Footer & Disclaimer
# ============================================================
# Force Streamlit to render a break and flush all open elements
st.write("")
#st.divider()
st.markdown(
    """
<div style='text-align:left; color:gray; font-size:14px; line-height:1.5; margin-top:10px;'>
<b>Disclaimer:</b><br>
This dashboard is for <b>educational and informational purposes only</b> and 
<b>does not constitute financial advice</b>.<br>
Markets carry risk; always do your own research or consult a licensed financial advisor before investing.<br><br>
© 2025 <b>Raj Gupta</b> — <i>AI Stock Signals PRO v5.5.2</i>
</div>
    """,
    unsafe_allow_html=True,
)


