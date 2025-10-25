# stock_dashboard_streamlit_pro_v5_4.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests, feedparser
from io import StringIO
from datetime import datetime, timedelta

# --- ML (optional) ---
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(page_title="AI Stock Signals — PRO v5.4", layout="wide")
st.title("🧠📈 AI Stock Signals — PRO v5.4")
st.caption("Technicals + macro + news + analyst pulse • Patterns • Short-term AI forecast • Adaptive DCA")

# ============================================================
# Inputs
# ============================================================
c1, c2, c3 = st.columns([2,2,3])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Simulation amount ($)", min_value=500, max_value=50_000, step=500, value=10_000)

# ============================================================
# Helpers
# ============================================================
def _fred_csv(series_id: str) -> pd.DataFrame | None:
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.rename(columns={series_id: "value"})
        df = df.dropna(subset=["DATE", "value"])
        df = df[df["value"] != "."]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna()
    except Exception:
        return None

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
    macro = {"vix_last": None, "spx_trend": None, "spx_5d_vs_20d": None,
             "cpi_yoy": None, "cpi_date": None, "unemp_rate": None, "unemp_date": None,
             "dgs10": None}
    # VIX & S&P
    try:
        vix = yf.download("^VIX", period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"]
        vix = vix.dropna()
        macro["vix_last"] = float(vix.iloc[-1]) if len(vix) else None
    except Exception:
        pass
    try:
        spx = yf.download("^GSPC", period="6mo", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
        if len(spx) >= 20:
            ma5, ma20 = spx.rolling(5).mean(), spx.rolling(20).mean()
            macro["spx_trend"] = "Bullish" if float(ma5.iloc[-1]) > float(ma20.iloc[-1]) else "Bearish"
            macro["spx_5d_vs_20d"] = round(((float(ma5.iloc[-1])-float(ma20.iloc[-1]))/float(ma20.iloc[-1]))*100, 2)
    except Exception:
        pass
    # CPI (YoY from index)
    cpi = _fred_csv("CPIAUCSL")
    if cpi is not None and len(cpi) > 13:
        cpi = cpi.sort_values("DATE")
        last, prev12 = cpi.iloc[-1]["value"], cpi.iloc[-13]["value"]
        if pd.notna(last) and pd.notna(prev12) and prev12:
            macro["cpi_yoy"] = round(((last/prev12)-1)*100, 2)
            macro["cpi_date"] = cpi.iloc[-1]["DATE"].date().isoformat()
    # Unemployment
    un = _fred_csv("UNRATE")
    if un is not None and len(un):
        un = un.sort_values("DATE")
        macro["unemp_rate"] = round(float(un.iloc[-1]["value"]), 2)
        macro["unemp_date"] = un.iloc[-1]["DATE"].date().isoformat()
    # 10Y Yield
    dgs10 = _fred_csv("DGS10")
    if dgs10 is not None and len(dgs10):
        dgs10 = dgs10.sort_values("DATE")
        macro["dgs10"] = round(float(dgs10.iloc[-1]["value"]), 2)
    return macro

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    # MAs
    out["MA20"]  = c.rolling(20, min_periods=1).mean()
    out["MA50"]  = c.rolling(50, min_periods=1).mean()
    out["MA200"] = c.rolling(200, min_periods=1).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()
    # RSI (Wilder)
    delta = c.diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out["RSI"] = (100 - (100/(1+rs))).fillna(50)
    # MACD
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    # Bollinger
    # Bollinger
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std(ddof=0)
    bb_up  = bb_mid + 2*bb_std
    bb_low = bb_mid - 2*bb_std
    out["BB_Up"], out["BB_Low"] = bb_up, bb_low
    out["BB_Width"] = (bb_up - bb_low) / c.replace(0, np.nan)

    # ATR
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
    # Volume spike
    vol_ma = v.rolling(20, min_periods=1).mean()
    out["Vol_Spike"] = (v > 2*vol_ma).astype(int)
    out["Close"] = c
    return out.bfill().ffill()

# ============================================================
# Patterns (heuristics)
# ============================================================
def detect_golden_death_cross(ind: pd.DataFrame):
    gc = dc = False
    if len(ind) < 205:
        return gc, dc
    ma50 = ind["MA50"].values
    ma200= ind["MA200"].values
    # last 10 days crossover check
    for i in range(-10, -1):
        if ma50[i-1] < ma200[i-1] and ma50[i] >= ma200[i]:
            gc = True
        if ma50[i-1] > ma200[i-1] and ma50[i] <= ma200[i]:
            dc = True
    return gc, dc

def detect_double_bottom(ind: pd.DataFrame):
    # two local minima within ~5% over last ~120 bars
    if len(ind) < 60: return False
    c = ind["Close"].tail(180)
    locs = ( (c.shift(1) > c) & (c.shift(-1) > c) )
    lows = c[locs]
    if len(lows) < 2: return False
    lows = lows.sort_values().head(4).sort_index()
    # pick two recent
    lows = lows.tail(2)
    if len(lows) < 2: return False
    v1, v2 = lows.iloc[0], lows.iloc[1]
    return (abs(v1 - v2)/((v1+v2)/2) < 0.05)

def detect_cup_handle(ind: pd.DataFrame):
    # super-simple: rounded bottom + small pullback then recover near prior high
    if len(ind) < 120: return False
    c = ind["Close"].tail(200)
    m = c.rolling(20, min_periods=1).mean()
    # bowl curvature: mean lower in middle than edges
    mid = m.iloc[len(m)//2]
    left, right = m.iloc[0], m.iloc[-1]
    bowl = (mid < left*0.98) and (mid < right*0.98)
    # handle: last 20 bars mild dip vs last 60 high
    recent = c.tail(60); last20 = c.tail(20)
    handle = (last20.min() > recent.max()*0.85)
    return bool(bowl and handle)

def compression_breakout_risk(ind: pd.DataFrame):
    # BB width percentile over last 120 bars
    if len(ind) < 40: return "Normal"
    ww = ind["BB_Width"].tail(120).dropna()
    if ww.empty: return "Normal"
    curr = ww.iloc[-1]
    pct = (ww < curr).mean()  # percentile
    if curr < ww.quantile(0.20):
        return "Compression (Breakout Risk)"
    return "Normal"

# ============================================================
# News + Sentiment
# ============================================================
def fetch_news_and_sentiment(ticker: str):
    analyzer = SentimentIntensityAnalyzer()
    headlines, scores = [], []
    key = st.secrets.get("NEWSAPI_KEY") if hasattr(st, "secrets") else None
    if key:
        try:
            url = ("https://newsapi.org/v2/everything?"
                   f"q={ticker}&language=en&sortBy=publishedAt&pageSize=10&apiKey={key}")
            r = requests.get(url, timeout=10)
            if r.ok:
                data = r.json()
                for a in data.get("articles", [])[:10]:
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

@st.cache_data(ttl=86400)
def analyst_pulse(ticker: str):
    try:
        t = yf.Ticker(ticker)
        rec = getattr(t, "recommendations", None)
        if rec is None or rec.empty:
            return {"buy_ratio": None, "samples": 0}
        df = rec.tail(200).copy()
        df.columns = [c.lower() for c in df.columns]
        actions = df.get("action")
        if actions is None:
            return {"buy_ratio": None, "samples": 0}
        actions = actions.astype(str).str.lower()
        ups = actions.str.contains("upgrade").sum()
        downs = actions.str.contains("downgrade").sum()
        total = ups + downs
        buy_ratio = (ups / total) if total > 0 else None
        return {"buy_ratio": buy_ratio, "samples": int(total)}
    except Exception:
        return {"buy_ratio": None, "samples": 0}

def market_confidence(sentiment: float, buy_ratio: float | None):
    sent_norm = (sentiment + 1) / 2   # 0..1
    if buy_ratio is None:
        conf = 0.6 * sent_norm + 0.4 * 0.5
        label = "Based on sentiment"
    else:
        conf = 0.6 * sent_norm + 0.4 * buy_ratio
        label = "Sentiment + analyst pulse"
    return int(round(conf * 100)), label

# ============================================================
# Signal logic
# ============================================================
def generate_signal(ind: pd.DataFrame, sentiment: float, horizon: str):
    last = ind.iloc[-1]
    score = 0.0
    # Trend
    if last["MA20"] > last["MA50"]:  score += 1
    if last["MA50"] > last["MA200"]: score += 1
    if last["ADX"] > 25:             score += 1
    # Momentum
    if last["RSI"] < 30: score += 1.2
    elif last["RSI"] > 70: score -= 1.2
    # MACD
    if last["MACD"] > last["MACD_Signal"]: score += 1
    else: score -= 1
    # Extremes
    if last["Close"] < last["BB_Low"]: score += 0.8
    elif last["Close"] > last["BB_Up"]: score -= 0.8
    # Volume
    if last["Vol_Spike"]: score += 0.3
    # News
    score += float(np.clip(sentiment, -0.8, 0.8))
    # Thresholds
    th_buy, th_sell = (2.5, -2.0) if "Short" in horizon else (3.5, -2.5)
    if score >= th_buy:  return "BUY", "green", round(score, 2)
    if score <= th_sell: return "SELL", "red",  round(score, 2)
    return "HOLD", "orange", round(score, 2)

def explain_signal(ind: pd.DataFrame, sentiment: float, decision: str) -> str:
    last = ind.iloc[-1]
    reasons = []
    reasons.append("MA trend up" if last["MA50"] > last["MA200"] else "MA trend down")
    reasons.append("MACD bullish" if last["MACD"] > last["MACD_Signal"] else "MACD bearish")
    if last["RSI"] < 35: reasons.append("RSI low → oversold")
    elif last["RSI"] > 65: reasons.append("RSI high → overbought")
    if last["Close"] < last["BB_Low"]: reasons.append("Below lower Bollinger (extreme)")
    elif last["Close"] > last["BB_Up"]: reasons.append("Above upper Bollinger (stretched)")
    if sentiment > 0.1: reasons.append("Positive news sentiment")
    elif sentiment < -0.1: reasons.append("Negative news sentiment")
    return f"**Why {decision}:** " + ", ".join(reasons)

def confidence_from_score(score: float) -> float:
    return float(min(abs(score) / 5.0, 1.0))

# ============================================================
# Backtest preview (simple RSI swing)
# ============================================================
def backtest_preview(df: pd.DataFrame, ind: pd.DataFrame) -> float:
    sig = (ind["RSI"] < 35).astype(int) - (ind["RSI"] > 65).astype(int)
    nxt = df["Close"].shift(-5)
    ret = (nxt - df["Close"]) / df["Close"]
    sig = np.ravel(sig.values); ret = np.ravel(ret.values)
    mask = sig != 0
    if mask.sum() < 10: return 0.0
    acc = np.mean(np.sign(ret[mask]) == sig[mask])
    return round(acc * 100, 1)

# ============================================================
# Adaptive DCA with partial take-profit + equity curve
# ============================================================
def adaptive_dca_simulator(df: pd.DataFrame, ind: pd.DataFrame, cash_start: float):
    df, ind = df.align(ind, join="inner", axis=0)
    cash = float(cash_start)
    shares = 0.0
    equity_curve = []
    trades = []
    peak_equity = cash_start
    halt_buys = False
    for dt in df.index:
        price = float(df.loc[dt, "Close"])
        rsi   = float(ind.loc[dt, "RSI"])
        macd  = float(ind.loc[dt, "MACD"])
        macds = float(ind.loc[dt, "MACD_Signal"])
        ma20  = float(ind.loc[dt, "MA20"])
        ma50  = float(ind.loc[dt, "MA50"])
        bb_low= float(ind.loc[dt, "BB_Low"])
        atr   = float(ind.loc[dt, "ATR"])

        # buys (adaptive)
        if not halt_buys:
            momentum_buy = (macd > macds and ma20 > ma50)
            oversold_buy = (rsi < 45) or (price < bb_low)
            alloc = 0.0
            if momentum_buy or oversold_buy:
                if rsi < 25:   alloc = 0.30
                elif rsi < 35: alloc = 0.20
                elif rsi < 45: alloc = 0.10
            invest = cash * alloc
            if invest > 0:
                buy_shares = invest / price
                shares += buy_shares
                cash   -= invest
                trades.append({"date": dt.strftime("%Y-%m-%d"), "side": "BUY",
                               "price": round(price,2), "invested": round(invest,2),
                               "shares": round(buy_shares,6)})

        # partial take-profit (2*ATR from latest close proxy)
        target_price = float(ind["Close"].iloc[-1] + 2*ind["ATR"].iloc[-1])
        if shares > 0 and price >= target_price:
            sell_shares = shares * 0.20
            proceeds = sell_shares * price
            shares -= sell_shares
            cash   += proceeds
            trades.append({"date": dt.strftime("%Y-%m-%d"), "side": "SELL",
                           "price": round(price,2), "invested": -round(proceeds,2),
                           "shares": -round(sell_shares,6)})

        equity = float(shares * price + cash)
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        dd_pct = (equity - peak_equity) / (peak_equity if peak_equity else 1)
        if dd_pct < -0.30:
            halt_buys = True

    final_value = shares * df["Close"].iloc[-1] + cash
    invested = sum(max(0.0, t["invested"]) for t in trades)
    pnl = final_value - invested
    roi_pct = (pnl / invested * 100) if invested > 0 else 0.0

    ec = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(ec) if ec.size else np.array([0])
    dd = (ec - running_max) / np.where(running_max == 0, 1, running_max)
    max_dd = float(np.min(dd)) if dd.size else 0.0

    return dict(
        final_value=float(final_value),
        total_invested=float(invested),
        roi_pct=float(roi_pct),
        max_drawdown_pct=round(100*max_dd, 2),
        trades=pd.DataFrame(trades),
        equity_curve=pd.Series(ec, index=df.index) if ec.size else pd.Series([], dtype=float)
    )

# ============================================================
# AI Forecast (next 5 days direction) — RandomForest (safe fallback)
# ============================================================
def build_features(df: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["ret_1"] = df["Close"].pct_change()
    X["ret_5"] = df["Close"].pct_change(5)
    X["ret_10"] = df["Close"].pct_change(10)
    for col in ["RSI","MACD","MACD_Hist","ADX","BB_Width","MA20","MA50","MA200","ATR"]:
        X[col] = ind[col]
    # MA relations
    X["ma20_above_50"] = (ind["MA20"] > ind["MA50"]).astype(int)
    X["ma50_above_200"] = (ind["MA50"] > ind["MA200"]).astype(int)
    return X.replace([np.inf, -np.inf], np.nan).dropna()

def target_y(df: pd.DataFrame) -> pd.Series:
    fwd_5 = df["Close"].shift(-5) / df["Close"] - 1.0
    y = np.sign(fwd_5)  # -1, 0, +1
    y = y.replace(0, 0)  # keep 0 as neutral
    return y

def ai_forecast(df: pd.DataFrame, ind: pd.DataFrame):
    if not SKLEARN_OK:
        return None  # gracefully skip if sklearn unavailable
    X = build_features(df, ind)
    y = target_y(df).reindex(X.index)
    # drop rows with NaN in y
    mask = y.notna()
    X, y = X[mask], y[mask]
    # simplify into up/down (0 for flat) for probability
    y = (y > 0).astype(int)
    y_arr = np.array(y, dtype=float)
    if len(X) < 200 or np.nansum(y_arr) == 0 or np.nansum(y_arr) == len(y_arr):
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X.iloc[:-5], y.iloc[:-5], test_size=0.25, shuffle=False
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, min_samples_leaf=4
        )
        rf.fit(X_train, y_train)
        acc = float(rf.score(X_test, y_test))
        # probability for last available row
        p_up = float(rf.predict_proba(X.iloc[[-1]])[0,1])
        return {"prob_up_5d": p_up, "test_acc": acc}
    except Exception:
        return None

# ============================================================
# Plot
# ============================================================
def plot_dashboard(ind: pd.DataFrame, ticker: str, show_zones=True):
    last = ind.iloc[-1]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05, subplot_titles=("Price / MAs / Bollinger + Zones", "MACD", "RSI")
    )
    # Price & bands
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color="blue")), 1, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA50"],  name="MA50",  line=dict(color="orange")), 1, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MA200"], name="MA200", line=dict(color="green")), 1, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Up"],  name="BB Upper", line=dict(color="gray", dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Low"], name="BB Lower", line=dict(color="gray", dash="dot")), 1, 1)
    # Zones
    if show_zones:
        buy_zone   = last["Close"] - 1.5*last["ATR"]
        target_zone= last["Close"] + 2.0*last["ATR"]
        stop_loss  = last["Close"] - 2.5*last["ATR"]
        fig.add_hline(y=buy_zone,    line_color="dodgerblue", line_dash="dash", annotation_text="Buy Zone",   row=1, col=1)
        fig.add_hline(y=target_zone, line_color="seagreen",   line_dash="dash", annotation_text="Target",     row=1, col=1)
        fig.add_hline(y=stop_loss,   line_color="crimson",    line_dash="dash", annotation_text="Stop Loss",  row=1, col=1)
    # MACD
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"],        name="MACD",   line=dict(color="purple")), 2, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", line=dict(color="orange")), 2, 1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_Hist"], name="Hist", marker_color="gray", opacity=0.45), 2, 1)
    # RSI
    fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="teal")), 3, 1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.update_layout(height=820, template="plotly_white", title=f"{ticker} — Technical Dashboard")
    return fig

# ============================================================
# UI: Macro Context
# ============================================================
macro = fetch_macro()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("VIX (vol)", f"{macro['vix_last']:.2f}" if macro["vix_last"] is not None else "—")
m2.metric("S&P 5d vs 20d", f"{macro['spx_5d_vs_20d']:+.2f}%" if macro["spx_5d_vs_20d"] is not None else "—", macro["spx_trend"] or "")
m3.metric("CPI YoY", f"{macro['cpi_yoy']:.2f}%" if macro["cpi_yoy"] is not None else "—")
m4.metric("Unemployment", f"{macro['unemp_rate']:.2f}%" if macro["unemp_rate"] is not None else "—")
m5.metric("10Y Yield", f"{macro['dgs10']:.2f}%" if macro["dgs10"] is not None else "—")

# ============================================================
# MAIN FLOW
# ============================================================
if not ticker:
    st.stop()

df = fetch_prices(ticker, horizon)
if df is None:
    st.error("No data found. Try another symbol.")
    st.stop()

ind = compute_indicators(df)
headlines, news_sent = fetch_news_and_sentiment(ticker)
decision, color, score = generate_signal(ind, news_sent, horizon)

# Patterns
golden, death = detect_golden_death_cross(ind)
dbl_bottom = detect_double_bottom(ind)
cup_handle = detect_cup_handle(ind)
vol_state  = compression_breakout_risk(ind)

# Analyst pulse & confidence
pulse = analyst_pulse(ticker)
conf_pct, conf_label = market_confidence(news_sent, pulse["buy_ratio"])

# AI forecast
ai = ai_forecast(df, ind)

# Top metrics
last = ind.iloc[-1]
cA, cB, cC, cD, cE, cF = st.columns(6)
cA.metric("Price", f"${last['Close']:.2f}")
cB.metric("RSI (14)", f"{last['RSI']:.1f}")
cC.metric("MACD", f"{last['MACD']:.2f}")
cD.metric("ADX", f"{last['ADX']:.1f}")
cE.metric("ATR (14)", f"{last['ATR']:.2f}")
cF.metric("Analyst Pulse", f"{int(pulse['buy_ratio']*100)}% buys" if pulse["buy_ratio"] is not None else "—")

# Summary banner
st.markdown(
    f"### **Signal: `{decision}`** (Score {score:+.2f}, News {news_sent:+.2f}) — "
    + (f"**AI 5-day ↑ Prob:** {ai['prob_up_5d']*100:.0f}% • Test Acc {ai['test_acc']*100:.0f}%"
       if ai else "AI forecast unavailable (no sklearn / insufficient data)")
)
st.progress(conf_pct/100.0, text=f"Market Confidence {conf_pct}% — {conf_label}")

# Patterns badge row
pat = []
if golden: pat.append("🟢 Golden Cross")
if death:  pat.append("🔴 Death Cross")
if dbl_bottom: pat.append("🔵 Double Bottom")
if cup_handle: pat.append("🟣 Cup & Handle (heuristic)")
if vol_state != "Normal": pat.append("🟠 " + vol_state)
if pat:
    st.write("**Patterns:** " + " • ".join(pat))

# Chart
st.plotly_chart(plot_dashboard(ind, ticker, show_zones=True), use_container_width=True)

# Backtest preview + explanation
acc = backtest_preview(df, ind)
st.write(f"**Backtest Preview (RSI swing, 5-day horizon):** {acc:.1f}% accuracy")
st.markdown(explain_signal(ind, news_sent, decision))
st.progress(confidence_from_score(score))

# Adaptive DCA
st.markdown("## 💵 Adaptive DCA Simulator (long-only) — partial take-profit")
sim = adaptive_dca_simulator(df, ind, invest_amount)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Final Portfolio Value", f"${sim['final_value']:.2f}")
s2.metric("Total Invested", f"${sim['total_invested']:.2f}")
s3.metric("ROI", f"{sim['roi_pct']:.1f}%")
s4.metric("Max Drawdown", f"{sim['max_drawdown_pct']:.1f}%")

if not sim["trades"].empty:
    with st.expander("🔎 Trades Executed"):
        st.dataframe(sim["trades"], use_container_width=True)
if isinstance(sim.get("equity_curve"), pd.Series) and not sim["equity_curve"].empty:
    ec_fig = go.Figure()
    ec_fig.add_trace(go.Scatter(x=sim["equity_curve"].index, y=sim["equity_curve"].values, name="Equity"))
    ec_fig.update_layout(height=300, template="plotly_white", title="Simulation Equity Curve")
    st.plotly_chart(ec_fig, use_container_width=True)

# News
with st.expander("🗞️ Latest Headlines"):
    if not headlines:
        st.write("No headlines available.")
    else:
        for h in headlines:
            title = h["title"]; url = h["url"]
            src = h.get("source",""); pub = h.get("published","")
            nice = pub[:10] if pub else ""
            st.markdown(f"- [{title}]({url}) — *{src}* {('• '+nice) if nice else ''}")

# Learn
with st.expander("📘 Learn: Indicators, Patterns & Strategy"):
    st.markdown("""
**RSI** — <30 oversold, >70 overbought.  
**MACD** — momentum/trend; crossovers can signal shifts.  
**Bollinger Bands** — ±2σ of 20-day mean; outside bands = extremes; narrow width → compression → potential breakout.  
**ADX** — trend strength (>25 often strong).  
**ATR** — volatility; use to set dynamic stops/targets.  
**Golden/Death Cross** — MA50 crossing MA200.  
**Cup & Handle** — rounded base + shallow pullback; breakout above rim on volume.  
**Double Bottom** — two similar lows; breakout over the midpoint.  
**Adaptive DCA** — invest more when RSI is deeply oversold; partial take-profit at targets.
""")

# Footer / Disclaimer
st.markdown("---")
st.caption("© 2025 Raj Gupta — AI Stock Signals PRO v5.4 • Educational use only • Not financial advice")
