# stock_dashboard_streamlit_pro_v5_3_3.py
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

# =============================== #
#   STREAMLIT CONFIG & HEADER     #
# =============================== #
st.set_page_config(page_title="AI Stock Signals — PRO 5.3.3", layout="wide")
st.title("🧠📈 AI Stock Signals — PRO 5.3.3")
st.caption("Technicals + macro + news sentiment + analyst pulse • DCA simulator • Backtest preview • Educational notes")

# =============================== #
#             INPUTS              #
# =============================== #
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Adaptive DCA total budget ($)", min_value=500, max_value=50_000, step=500, value=10_000)

# =============================== #
#          DATA FETCHERS          #
# =============================== #
@st.cache_data(ttl=7200)
def fetch_prices(ticker: str, horizon: str):
    """Download OHLCV (auto-adjusted) with horizon-aware period."""
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
def _fred_csv_to_df(url: str):
    """Load a FRED CSV into a clean dataframe with DATE,value."""
    try:
        r = requests.get(url, timeout=12)
        if not r.ok:
            return None
        df = pd.read_csv(StringIO(r.text))
        # try graph format (fredgraph.csv?id=SERIES) or downloaddata (DATE, SERIES)
        if "DATE" not in df.columns:
            return None
        # Identify the value column
        val_col = None
        for c in df.columns:
            if c != "DATE":
                val_col = c
                break
        if not val_col:
            return None
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.rename(columns={val_col: "value"}).dropna()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("DATE")
        return df
    except Exception:
        return None

@st.cache_data(ttl=86400)
def fetch_macro():
    """
    Macro dashboard with robust fallbacks:
      - VIX via yfinance
      - S&P trend via yfinance (^GSPC MA5 vs MA20)
      - CPI YoY: FRED (primary) + alternate FRED CSV path
      - Unemployment: FRED series UNRATE with same fallback
    """
    macro = {}

    # --- VIX ---
    try:
        vix = yf.download("^VIX", period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"]
        macro["vix_last"] = float(vix.dropna().iloc[-1]) if not vix.empty else None
    except Exception:
        macro["vix_last"] = None

    # --- S&P trend: 5d vs 20d ---
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

    # --- CPI YoY (FRED primary + fallback path) ---
    cpi_df = None
    for url in [
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
        "https://fred.stlouisfed.org/series/CPIAUCSL/downloaddata/CPIAUCSL.csv",
    ]:
        cpi_df = _fred_csv_to_df(url)
        if cpi_df is not None:
            break

    if cpi_df is not None and len(cpi_df) > 13:
        last, prev12 = cpi_df.iloc[-1]["value"], cpi_df.iloc[-13]["value"]
        macro["cpi_yoy"] = round(((last / prev12) - 1) * 100, 2)
        macro["cpi_date"] = cpi_df.iloc[-1]["DATE"].date().isoformat()
    else:
        macro["cpi_yoy"], macro["cpi_date"] = None, None

    # --- Unemployment (UNRATE) ---
    un_df = None
    for url in [
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
        "https://fred.stlouisfed.org/series/UNRATE/downloaddata/UNRATE.csv",
    ]:
        un_df = _fred_csv_to_df(url)
        if un_df is not None:
            break

    if un_df is not None and len(un_df) > 0:
        macro["unemp_rate"] = round(float(un_df.iloc[-1]["value"]), 2)
        macro["unemp_date"] = un_df.iloc[-1]["DATE"].date().isoformat()
    else:
        macro["unemp_rate"], macro["unemp_date"] = None, None

    # --- Context/explanations ---
    ctx = []
    if macro.get("vix_last") is not None:
        if macro["vix_last"] >= 25:
            ctx.append("VIX elevated → broader risk-off conditions.")
        elif macro["vix_last"] <= 14:
            ctx.append("VIX subdued → supportive for risk assets.")
    if macro.get("spx_trend") == "Bullish":
        ctx.append("S&P short-term MA above medium-term MA → constructive tone.")
    elif macro.get("spx_trend") == "Bearish":
        ctx.append("S&P short-term MA below medium-term MA → cautious tone.")
    if macro.get("cpi_yoy") is not None:
        if macro["cpi_yoy"] > 4:
            ctx.append("Inflation elevated → rate-sensitive sectors may lag.")
        elif macro["cpi_yoy"] < 3:
            ctx.append("Inflation moderating → supportive for multiples.")
    macro["context"] = " ".join(ctx) if ctx else None

    return macro

# =============================== #
#           ANALYST PULSE         #
# =============================== #
@st.cache_data(ttl=86400)
def analyst_pulse(ticker: str):
    try:
        t = yf.Ticker(ticker)
        # New-style summary
        if getattr(t, "recommendations_summary", None) is not None:
            df = t.recommendations_summary
            if isinstance(df, pd.DataFrame) and not df.empty:
                recs = df.iloc[0].to_dict()
                buy = int(recs.get("strongBuy", 0) + recs.get("buy", 0))
                hold = int(recs.get("hold", 0))
                sell = int(recs.get("strongSell", 0) + recs.get("sell", 0))
                total = buy + hold + sell
                return {"buy_ratio": (buy / total) if total > 0 else None, "samples": total}
        # Legacy table fallback
        rec = getattr(t, "recommendations", None)
        if rec is not None and not rec.empty:
            actions = rec.tail(200)["Action"].astype(str).str.lower()
            ups = int(actions.str.contains("upgrade").sum())
            downs = int(actions.str.contains("downgrade").sum())
            total = ups + downs
            return {"buy_ratio": (ups / total) if total > 0 else None, "samples": total}
    except Exception:
        pass
    return {"buy_ratio": None, "samples": 0}

def market_confidence(sentiment: float, buy_ratio: float | None):
    """Blend headline sentiment (-1..+1) with analyst buy ratio (0..1) → 0..100%"""
    sent_norm = (sentiment + 1) / 2  # map to 0..1
    if buy_ratio is None:
        conf = 0.65 * sent_norm + 0.35 * 0.5
        label = "Based on sentiment only"
    else:
        conf = 0.6 * sent_norm + 0.4 * buy_ratio
        label = "Sentiment + analyst pulse"
    return int(round(conf * 100)), label

# =============================== #
#          INDICATOR ENGINE       #
# =============================== #
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

# =============================== #
#       NEWS + SENTIMENT          #
# =============================== #
def fetch_news_and_sentiment(ticker: str):
    analyzer = SentimentIntensityAnalyzer()
    scores, headlines = [], []
    api_key = st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None

    # 1) NewsAPI (best: includes URLs)
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
                    url_a = a.get("url")
                    if title and url_a:
                        headlines.append({"title": title, "source": src or "News", "url": url_a})
                        scores.append(analyzer.polarity_scores(title)["compound"])
        except Exception:
            pass

    # 2) RSS fallback (Yahoo + CNBC) with links if available
    if not headlines:
        feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]
        for f in feeds:
            try:
                d = feedparser.parse(f)
                for e in d.entries[:6]:
                    title = getattr(e, "title", None)
                    link = getattr(e, "link", None)
                    if not title or not link:
                        continue
                    src = "Yahoo" if "yahoo" in f else "CNBC"
                    headlines.append({"title": title, "source": src, "url": link})
                    scores.append(analyzer.polarity_scores(title)["compound"])
            except Exception:
                continue

    sentiment = float(np.mean(scores)) if scores else 0.0
    return headlines[:6], sentiment

# =============================== #
#        SIGNAL GENERATOR         #
# =============================== #
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

    # Mean reversion
    if last["Close"] < last["BB_Low"]: score += 0.6
    elif last["Close"] > last["BB_Up"]: score -= 0.6

    # News sentiment (bounded)
    score += float(np.clip(sentiment, -0.8, 0.8))

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

# =============================== #
#        BACKTEST PREVIEW         #
# =============================== #
def backtest_preview(ind: pd.DataFrame) -> float:
    """Very quick swing sanity check: RSI <35 buy / >65 sell vs 5-day forward return direction."""
    test = ind.copy()
    test["Sig"] = 0
    test.loc[test["RSI"] < 35, "Sig"] = 1
    test.loc[test["RSI"] > 65, "Sig"] = -1
    test["Next"] = test["Close"].shift(-5)
    test["Ret"] = (test["Next"] - test["Close"]) / test["Close"]
    mask = test["Sig"] != 0
    if mask.sum() < 10:
        return 0.0
    accuracy = (np.sign(test.loc[mask, "Ret"]) == test.loc[mask, "Sig"]).mean()
    return float(round(accuracy * 100, 1))

# =============================== #
#     ADAPTIVE DCA SIMULATOR      #
# =============================== #
def adaptive_dca_simulator(df: pd.DataFrame, ind: pd.DataFrame, total_budget: float):
    """
    Invests over the most recent ~12 months:
      - Base monthly DCA (equal chunks)
      - Extra dip buys when RSI<35 or price < MA50*(1-3%)
    Stops when budget is exhausted. Returns trade log & summary.
    """
    if len(df) < 60:
        return pd.DataFrame(), {"shares": 0.0, "cash": total_budget, "final_value": total_budget}

    # Build monthly points = first trading day of each month in window
    df_month = df.resample("MS").first().dropna()
    months = df_month.index[-12:] if len(df_month) >= 12 else df_month.index
    base_chunk = total_budget / max(len(months), 1)

    cash = total_budget
    shares = 0.0
    trades = []

    # helper to execute a buy
    def buy(dt, price, usd, reason):
        nonlocal cash, shares, trades
        if price <= 0 or usd <= 0 or cash <= 1e-6:
            return
        usd = min(usd, cash)
        qty = usd / price
        cash -= usd
        shares += qty
        trades.append({"date": dt, "type": reason, "price": float(price), "usd": float(usd), "shares": float(qty)})

    # Iterate days; invest base on month starts; add dip buys
    ma50 = ind["MA50"]
    rsi = ind["RSI"]

    month_starts = set(pd.to_datetime(months).date)
    for dt, row in df.iterrows():
        price = row["Close"]
        date_only = dt.date()

        # base monthly DCA on first trading day of month
        if date_only in month_starts and cash > 0:
            buy(dt, price, base_chunk, "DCA")

        # dip buys (small bites) — 3% of initial budget each time
        if cash > 0:
            cond_rsi = rsi.loc[dt] < 35 if dt in rsi.index else False
            cond_ma = (price < ma50.loc[dt] * 0.97) if dt in ma50.index and ma50.loc[dt] > 0 else False
            if cond_rsi or cond_ma:
                buy(dt, price, total_budget * 0.03, "Dip")

        # stop if cash largely depleted
        if cash < total_budget * 0.02:
            break

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.sort_values("date")

    last_price = df["Close"].iloc[-1]
    final_value = cash + shares * last_price
    summary = {"shares": float(shares), "cash": float(cash), "final_value": float(final_value)}
    return trades_df, summary

# =============================== #
#              UI                 #
# =============================== #
macro = fetch_macro()
m1, m2, m3, m4 = st.columns(4)
m1.metric("VIX", f"{macro['vix_last']:.2f}" if macro['vix_last'] else "—")
m2.metric("S&P 5d vs 20d", f"{macro['spx_5d_vs_20d']:+.2f}%" if macro['spx_5d_vs_20d'] is not None else "—",
          macro['spx_trend'] or "")
m3.metric("CPI YoY", f"{macro['cpi_yoy']:.2f}%" if macro['cpi_yoy'] is not None else "—",
          macro.get("cpi_date", ""))
m4.metric("Unemployment", f"{macro['unemp_rate']:.2f}%" if macro['unemp_rate'] is not None else "—",
          macro.get("unemp_date", ""))

if macro.get("context"):
    st.info(macro["context"])

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

# --- Top metrics ---
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

# --- Chart ---
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

st.plotly_chart(plot_indicators(ind, ticker), use_container_width=True)

# --- Backtest preview ---
acc = backtest_preview(ind)
st.write(f"**Backtest Preview (RSI swing, 5-day horizon):** {acc:.1f}% directional accuracy")

# --- Why explanation ---
st.markdown(explain_signal(ind, news_sent, decision))

# --- Headlines (clickable) ---
st.markdown("#### 🗞️ Latest Headlines")
if headlines:
    for h in headlines:
        st.markdown(f"- [{h['title']}]({h['url']}) — *{h['source']}*")
else:
    st.write("No recent headlines found.")

# --- Adaptive DCA simulator ---
st.markdown("### 💸 Adaptive DCA Simulator (12 months)")
trades_df, sim_summary = adaptive_dca_simulator(df, ind, float(invest_amount))
if trades_df.empty:
    st.info("Need more price history to simulate DCA (requires ~12 months).")
else:
    colS1, colS2, colS3 = st.columns(3)
    colS1.metric("Shares accumulated", f"{sim_summary['shares']:.4f}")
    colS2.metric("Cash remaining", f"${sim_summary['cash']:.2f}")
    colS3.metric("Final portfolio value", f"${sim_summary['final_value']:.2f}")
    st.dataframe(trades_df.assign(date=trades_df["date"].dt.strftime("%Y-%m-%d")))

# --- Education ---
with st.expander("📚 What the model looks at (learn the signals)"):
    st.markdown("""
- **Trend:** MA20/50/200 & **ADX** (trend strength). Uptrends = bias to BUY; downtrends = bias to SELL/avoid.
- **Momentum:** **MACD** crossovers and **RSI** extremes (oversold <30, overbought >70).
- **Mean reversion:** **Bollinger Bands** — price below lower band can bounce; above upper band can cool off.
- **Volatility:** **ATR** helps set realistic stops & targets.
- **News sentiment:** Recent headline tone nudges signals up/down.
- **Analyst pulse:** Blend of analyst recommendations adds confidence context.
- **DCA simulator:** Spreads buys monthly; adds small **dip buys** when RSI is weak or price is under MA50 by ~3%.
- **Backtest preview:** Quick sanity check using RSI swing → 5-day forward direction. Not a full backtest.
""")

st.markdown("---")
st.caption("© 2025 Raj Gupta — AI Stock Signals PRO 5.3.3 • Educational use only")
