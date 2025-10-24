# stock_dashboard_streamlit_pro_v5_3_full.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests, feedparser
from io import StringIO

# ----------------------------------------------------------
# Streamlit config
# ----------------------------------------------------------
st.set_page_config(page_title="AI Stock Signals — PRO 5.3 (Full)", layout="wide")
st.title("🧠📈 AI Stock Signals — PRO 5.3 (Full)")
st.caption("Technicals + macro + news sentiment + analyst pulse • Backtest preview • Educational notes")

# ----------------------------------------------------------
# Inputs
# ----------------------------------------------------------
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Simulation amount ($)", min_value=500, max_value=50_000, step=500, value=10_000)

# ----------------------------------------------------------
# Helpers: data fetching
# ----------------------------------------------------------
@st.cache_data(ttl=7200)
def fetch_prices(ticker: str, horizon: str):
    period = "6mo" if "Short" in horizon else "5y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df

@st.cache_data(ttl=86400)
def fetch_macro():
    """
    Macro dashboard: VIX, S&P trend (5d vs 20d), CPI YoY, Unemployment.
    Uses FRED CSV with a pandas-safe StringIO; explains context.
    """
    macro = {}

    # VIX
    try:
        vix = yf.download("^VIX", period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"]
        macro["vix_last"] = float(vix.dropna().iloc[-1]) if not vix.empty else None
    except Exception:
        macro["vix_last"] = None

    # S&P 500 trend (5d vs 20d)
    try:
        spx = yf.download("^GSPC", period="6mo", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
        if spx.empty or len(spx) < 20:
            macro["spx_trend"], macro["spx_5d_vs_20d"] = None, None
        else:
            ma5 = spx.rolling(5).mean()
            ma20 = spx.rolling(20).mean()
            ma5_last, ma20_last = float(ma5.iloc[-1]), float(ma20.iloc[-1])
            macro["spx_trend"] = "Bullish" if ma5_last > ma20_last else "Bearish"
            macro["spx_5d_vs_20d"] = round(((ma5_last - ma20_last) / ma20_last) * 100, 2)
    except Exception:
        macro["spx_trend"], macro["spx_5d_vs_20d"] = None, None

    # FRED CSV loader (pandas>=2.2 safe)
    def fred_csv_last(series_id: str):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            r = requests.get(url, timeout=10)
            if not r.ok:
                return None
            df = pd.read_csv(StringIO(r.text))
            if series_id not in df.columns:
                return None
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.rename(columns={series_id: "value"}).dropna()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna()
        except Exception:
            return None

    # CPI YoY
    try:
        cpi_df = fred_csv_last("CPIAUCSL")
        if cpi_df is not None and len(cpi_df) > 13:
            last, prev12 = cpi_df.iloc[-1]["value"], cpi_df.iloc[-13]["value"]
            macro["cpi_yoy"] = round(((last / prev12) - 1) * 100, 2)
            macro["cpi_date"] = cpi_df.iloc[-1]["DATE"].date().isoformat()
        else:
            macro["cpi_yoy"], macro["cpi_date"] = None, None
    except Exception:
        macro["cpi_yoy"], macro["cpi_date"] = None, None

    # Unemployment rate
    try:
        un_df = fred_csv_last("UNRATE")
        if un_df is not None and len(un_df) > 0:
            macro["unemp_rate"] = round(float(un_df.iloc[-1]["value"]), 2)
            macro["unemp_date"] = un_df.iloc[-1]["DATE"].date().isoformat()
        else:
            macro["unemp_rate"], macro["unemp_date"] = None, None
    except Exception:
        macro["unemp_rate"], macro["unemp_date"] = None, None

    # Context/explanations
    ctx = []
    if macro.get("vix_last") is not None:
        if macro["vix_last"] >= 25:
            ctx.append("VIX elevated → broader risk-off conditions.")
        elif macro["vix_last"] <= 14:
            ctx.append("VIX subdued → supportive backdrop for risk assets.")
    if macro.get("spx_trend") == "Bullish":
        ctx.append("S&P short-term MA above medium-term MA → bullish market tone.")
    elif macro.get("spx_trend") == "Bearish":
        ctx.append("S&P short-term MA below medium-term MA → cautious tone.")
    if macro.get("cpi_yoy") is not None:
        if macro["cpi_yoy"] > 4:
            ctx.append("Inflation elevated → rate-sensitive sectors may lag.")
        elif macro["cpi_yoy"] < 3:
            ctx.append("Inflation moderating → supportive for multiples.")
    macro["context"] = " ".join(ctx) if ctx else None

    return macro

# ----------------------------------------------------------
# Analyst Pulse (yfinance-based, with fallbacks)
# ----------------------------------------------------------
@st.cache_data(ttl=86400)
def analyst_pulse(ticker: str):
    try:
        t = yf.Ticker(ticker)
        # Newer summary table
        if hasattr(t, "recommendations_summary") and t.recommendations_summary is not None:
            df = t.recommendations_summary
            if isinstance(df, pd.DataFrame) and not df.empty:
                recs = df.iloc[0].to_dict()
                buy = recs.get("strongBuy", 0) + recs.get("buy", 0)
                hold = recs.get("hold", 0)
                sell = recs.get("strongSell", 0) + recs.get("sell", 0)
                total = buy + hold + sell
                return {
                    "buy_ratio": (buy / total) if total > 0 else None,
                    "samples": int(total)
                }
        # Older table as fallback
        rec = getattr(t, "recommendations", None)
        if rec is not None and not rec.empty:
            actions = rec.tail(200)["Action"].astype(str).str.lower()
            ups = actions.str.contains("upgrade").sum()
            downs = actions.str.contains("downgrade").sum()
            total = ups + downs
            return {"buy_ratio": (ups / total) if total > 0 else None, "samples": int(total)}
    except Exception:
        pass
    return {"buy_ratio": None, "samples": 0}

def market_confidence(sentiment: float, buy_ratio: float | None):
    """
    Simple blend of news sentiment (-1..+1) and analyst buy ratio (0..1).
    """
    sent_norm = (sentiment + 1) / 2  # 0..1
    if buy_ratio is None:
        conf = 0.65 * sent_norm + 0.35 * 0.5
        label = "Based on sentiment only"
    else:
        conf = 0.6 * sent_norm + 0.4 * buy_ratio
        label = "Sentiment + analyst pulse"
    return int(round(conf * 100)), label

# ----------------------------------------------------------
# Indicators
# ----------------------------------------------------------
def compute_indicators(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # MAs
    out["MA20"] = c.rolling(20, min_periods=1).mean()
    out["MA50"] = c.rolling(50, min_periods=1).mean()
    out["MA200"] = c.rolling(200, min_periods=1).mean()

    # RSI (Wilder)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # Bollinger
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std(ddof=0)
    out["BB_Up"] = bb_mid + 2 * bb_std
    out["BB_Low"] = bb_mid - 2 * bb_std

    # ATR
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=1).mean()

    # ADX (flatten-safe)
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

    out["Close"] = c
    return out.bfill().ffill()

# ----------------------------------------------------------
# News & sentiment
# ----------------------------------------------------------
def fetch_news_and_sentiment(ticker: str):
    analyzer = SentimentIntensityAnalyzer()
    scores, headlines = [], []
    api_key = st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None

    # Prefer NewsAPI if key is present
    if api_key:
        try:
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={ticker}&language=en&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
            )
            r = requests.get(url, timeout=10)
            if r.ok:
                for a in r.json().get("articles", [])[:10]:
                    title = a.get("title", "")
                    src = a.get("source", {}).get("name", "")
                    if title:
                        headlines.append({"title": title, "source": src})
                        scores.append(analyzer.polarity_scores(title)["compound"])
        except Exception:
            pass

    # Fallback to RSS (Yahoo + CNBC)
    if not headlines:
        feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]
        for f in feeds:
            try:
                d = feedparser.parse(f)
                for e in d.entries[:5]:
                    title = getattr(e, "title", None)
                    if not title:
                        continue
                    src = "Yahoo" if "yahoo" in f else "CNBC"
                    headlines.append({"title": title, "source": src})
                    scores.append(analyzer.polarity_scores(title)["compound"])
            except Exception:
                continue

    sentiment = float(np.mean(scores)) if scores else 0.0
    return headlines[:5], sentiment

# ----------------------------------------------------------
# Signal engine
# ----------------------------------------------------------
def generate_signal(ind: pd.DataFrame, sentiment: float, horizon: str):
    last = ind.iloc[-1]
    score = 0.0

    # Trend
    if last["MA20"] > last["MA50"]: score += 0.8
    if last["MA50"] > last["MA200"]: score += 1.0
    if last["ADX"] > 25: score += 0.6

    # Momentum
    if last["RSI"] < 30: score += 1.2
    elif last["RSI"] > 70: score -= 1.2
    if last["MACD"] > last["MACD_Signal"]: score += 0.8
    else: score -= 0.8

    # Mean-reversion
    if last["Close"] < last["BB_Low"]: score += 0.6
    elif last["Close"] > last["BB_Up"]: score -= 0.6

    # News sentiment (bounded)
    score += float(np.clip(sentiment, -0.8, 0.8))

    # Thresholds by mode
    th_buy, th_sell = (2.2, -1.8) if "Short" in horizon else (3.2, -2.4)

    if score >= th_buy:
        return "BUY", "green", round(score, 2)
    if score <= th_sell:
        return "SELL", "red", round(score, 2)
    return "HOLD", "orange", round(score, 2)

def explain_signal(ind: pd.DataFrame, sentiment: float, decision: str) -> str:
    last = ind.iloc[-1]
    bullets = []
    bullets.append("MA50 > MA200 → longer-term uptrend" if last["MA50"] > last["MA200"] else "MA50 < MA200 → longer-term downtrend")
    bullets.append("MACD > signal → positive momentum" if last["MACD"] > last["MACD_Signal"] else "MACD < signal → weak momentum")
    if last["RSI"] < 35: bullets.append("RSI low → oversold zone")
    elif last["RSI"] > 65: bullets.append("RSI high → overbought risk")
    if last["Close"] < last["BB_Low"]: bullets.append("Price under lower Bollinger → potential rebound")
    elif last["Close"] > last["BB_Up"]: bullets.append("Price above upper Bollinger → stretched")
    if sentiment > 0.15: bullets.append("News tone: supportive")
    elif sentiment < -0.15: bullets.append("News tone: negative")
    head = f"**Why {decision}:** "
    return head + "; ".join(bullets) if bullets else head + "mixed signals."

# ----------------------------------------------------------
# Backtest preview (quick sanity check, not a full backtest)
# ----------------------------------------------------------
def backtest_preview(ind: pd.DataFrame) -> float:
    test = ind.copy()
    test["Sig"] = 0
    # simple swing rule
    test.loc[test["RSI"] < 35, "Sig"] = 1
    test.loc[test["RSI"] > 65, "Sig"] = -1
    test["Next"] = test["Close"].shift(-5)  # 5-day look-ahead
    test["Ret"] = (test["Next"] - test["Close"]) / test["Close"]
    mask = test["Sig"] != 0
    if mask.sum() < 10:
        return 0.0
    accuracy = (np.sign(test.loc[mask, "Ret"]) == test.loc[mask, "Sig"]).mean()
    return float(round(accuracy * 100, 1))

# ----------------------------------------------------------
# UI: Macro row
# ----------------------------------------------------------
macro = fetch_macro()
m1, m2, m3, m4 = st.columns(4)
m1.metric("VIX", f"{macro['vix_last']:.2f}" if macro['vix_last'] else "—")
m2.metric("S&P 5d vs 20d", f"{macro['spx_5d_vs_20d']:+.2f}%" if macro['spx_5d_vs_20d'] else "—",
          macro['spx_trend'] or "")
m3.metric("CPI YoY", f"{macro['cpi_yoy']:.2f}%" if macro['cpi_yoy'] is not None else "—",
          macro.get("cpi_date", ""))
m4.metric("Unemployment", f"{macro['unemp_rate']:.2f}%" if macro['unemp_rate'] is not None else "—",
          macro.get("unemp_date", ""))

if macro.get("context"):
    st.info(macro["context"])

# ----------------------------------------------------------
# Main ticker flow
# ----------------------------------------------------------
if not ticker:
    st.stop()

df = fetch_prices(ticker, horizon)
if df is None:
    st.error("No price data. Try another symbol.")
    st.stop()

ind = compute_indicators(df)
headlines, news_sent = fetch_news_and_sentiment(ticker)
decision, color, score = generate_signal(ind, news_sent, horizon)
pulse = analyst_pulse(ticker)
conf_pct, conf_label = market_confidence(news_sent, pulse["buy_ratio"])

# Summary metrics
last = ind.iloc[-1]
cA, cB, cC, cD, cE = st.columns(5)
cA.metric("Price", f"${last['Close']:.2f}")
cB.metric("RSI (14)", f"{last['RSI']:.1f}")
cC.metric("MACD", f"{last['MACD']:.2f}")
cD.metric("ADX", f"{last['ADX']:.1f}")
cE.metric("ATR (14)", f"{last['ATR']:.2f}")

st.markdown(f"### **Signal: {decision}** (Score {score:+.2f}, News {news_sent:+.2f}, Mode: {horizon})")
st.progress(conf_pct / 100.0, text=f"Market Confidence {conf_pct}% — {conf_label}")
st.write(f"Analyst Pulse: {'{:.0f}% buys'.format(pulse['buy_ratio']*100) if pulse['buy_ratio'] else 'No analyst data available'} "
         f"(n={pulse['samples']})")

# Plot
def plot_indicators(ind: pd.DataFrame, t: str):
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

    fig.update_layout(height=820, title=f"{t} — Technical Dashboard",
                      template="plotly_white", legend=dict(orientation="h", y=-0.08))
    return fig

fig = plot_indicators(ind, ticker)
st.plotly_chart(fig, use_container_width=True)

# Backtest preview
acc = backtest_preview(ind)
st.write(f"**Backtest Preview (RSI swing, 5-day horizon):** {acc:.1f}% directional accuracy")

# Why explanation
st.markdown(explain_signal(ind, news_sent, decision))

# Headlines
st.markdown("#### 🗞️ Latest Headlines")
if headlines:
    for h in headlines:
        st.markdown(f"- {h['title']} — *{h['source']}*")
else:
    st.write("No recent headlines found.")

# Education
with st.expander("📚 What the model looks at (learn the signals)"):
    st.markdown("""
- **Trend:** MA20/50/200 & **ADX** (trend strength). Uptrends = bias to BUY, downtrends = bias to SELL/avoid.
- **Momentum:** **MACD** crossovers and **RSI** extremes (oversold <30, overbought >70).
- **Mean reversion:** **Bollinger Bands** — price breaking below lower band can bounce; above upper band can cool off.
- **Volatility:** **ATR** guides realistic stop/targets; higher ATR = wider daily swings.
- **News sentiment:** Recent headline tone nudges signals up/down.
- **Analyst pulse:** If available, more buys than sells improves confidence.

**Backtest preview** is a quick sanity check (RSI swing). For investment decisions, use multiple confirmations and risk controls.
""")

st.markdown("---")
st.caption("© 2025 Raj Gupta — AI Stock Signals PRO 5.3 (Full) • Educational use only")
