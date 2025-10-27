# stock_dashboard_streamlit_pro_v5_5_1.py
# v5.5.1 – UX/Trust upgrades for v5.5 (targets, explanations, confidence, safe fallbacks)
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
st.set_page_config(page_title="AI Stock Signals — PRO v5.5.1", layout="wide")
st.title("🧠📊 AI Stock Signals — PRO v5.5.1")
st.caption("Technicals • Macro • News • Analyst • Hybrid AI Forecast • Adaptive DCA")

# ============= Inputs =============
c1, c2, c3 = st.columns([2,2,3])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Simulation amount ($)", min_value=500, max_value=50_000, step=500, value=10_000)

# ============= Data fetchers =============
@st.cache_data(ttl=7200)
def fetch_prices(ticker: str, horizon: str) -> pd.DataFrame | None:
    period = "6mo" if "Short" in horizon else "5y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return None
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    try: df.index = df.index.tz_localize(None)
    except Exception: pass
    return df

@st.cache_data(ttl=86400)
def fetch_macro():
    """Free macro with safe fallbacks so UI never shows blanks."""
    macro = {}

    # VIX
    try:
        vix = yf.download("^VIX", period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
        macro["vix_last"] = float(vix.iloc[-1]) if not vix.empty else None
    except Exception:
        macro["vix_last"] = None

    # S&P 5d vs 20d trend
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

    # CPI YoY + Unemployment from FRED CSV; graceful fallbacks
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

    # Safe fallbacks so metrics never blank
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

    # ADX (safe)
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

    # Extras for UI
    width = (out["BB_Up"] - out["BB_Low"]) / c.replace(0,np.nan)
    #out["BB_Width"] = width.fillna(0)
    # --- Bollinger width (safe 1-D Series)
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
    sent_norm = (sentiment + 1) / 2   # map -1..1 -> 0..1
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
    if last["BB_Width"] > 0.12: score -= 0.2  # very wide bands = volatile
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

df = fetch_prices(ticker, horizon)
if df is None:
    st.error("No data found. Try another symbol.")
    st.stop()

ind = compute_indicators(df)
macro = fetch_macro()
headlines, news_sent = fetch_news_and_sentiment(ticker)
decision, color, score = generate_signal(ind, news_sent, horizon)
pulse = analyst_pulse(ticker)
conf_overall = market_confidence(news_sent, pulse["buy_ratio"])
ai = ai_forecast(df, ind)

# Macro header
m1, m2, m3, m4 = st.columns(4)
m1.metric("VIX (volatility)", f"{macro['vix_last']:.2f}" if macro["vix_last"] is not None else "—")
m2.metric("S&P 5d vs 20d", f"{macro['spx_5d_vs_20d']:+.2f}%" if macro["spx_5d_vs_20d"] is not None else "—",
          macro["spx_trend"] or "")
m3.metric("CPI YoY", f"{macro['cpi_yoy']:.2f}%")
m4.metric("Unemployment", f"{macro['unemp_rate']:.2f}%")

# Snapshot metrics
last = ind.iloc[-1]
cA, cB, cC, cD, cE, cF = st.columns(6)
cA.metric("Price", f"${last['Close']:.2f}")
cB.metric("RSI (14)", f"{last['RSI']:.1f}")
cC.metric("MACD", f"{last['MACD']:.2f}")
cD.metric("ADX", f"{last['ADX']:.1f}")
cE.metric("ATR (14)", f"{last['ATR']:.2f}")
cF.metric("Analyst Pulse", f"{int(pulse['buy_ratio']*100)}% buys" if pulse["buy_ratio"] is not None else "—")

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
with st.expander("📘 Learn: Indicators, Patterns & AI Logic"):
    st.markdown("""
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
""")
# ============================================================
# 🧾 Footer & Disclaimer
# ============================================================
st.markdown("""
**Disclaimer:**  
This dashboard is for **educational and informational purposes only** and **does not constitute financial advice**.  
Markets carry risk; always do your own research or consult a licensed financial advisor before investing.  

© 2025 **Raj Gupta** — *AI Stock Signals PRO v5.5.1*
""")
