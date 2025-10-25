# stock_dashboard_streamlit_pro_v5_5.py
# ------------------------------------------------------------
# AI Stock Signals — PRO v5.5
# - Technicals + Macro + News Sentiment + Analyst Pulse
# - Hybrid AI Forecast (RandomForest + Markov Chain)
# - Adaptive DCA simulator with partial take-profit
# ------------------------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO
import requests, feedparser
from sklearn.ensemble import RandomForestRegressor

# =========================
# Page config & header
# =========================
st.set_page_config(page_title="AI Stock Signals — PRO v5.5", layout="wide")
st.title("🧠📊 AI Stock Signals — PRO v5.5")
st.caption("Technicals • Macro • News • Analyst • Hybrid AI Forecast • Adaptive DCA")

# =========================
# Inputs
# =========================
c1, c2, c3 = st.columns([2,2,3])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Simulation amount ($)", min_value=500, max_value=50_000, step=500, value=10_000)

if not ticker:
    st.stop()

# =========================
# Helpers: Data Fetch
# =========================
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
    """Free macro: CPI & Unemployment from FRED CSV (no key), VIX + S&P via yfinance."""
    macro = {}

    # --- VIX
    try:
        vix = yf.download("^VIX", period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"]
        macro["vix_last"] = float(vix.dropna().iloc[-1]) if not vix.empty else None
    except Exception:
        macro["vix_last"] = None

    # --- S&P short trend (5d vs 20d MA of ^GSPC)
    try:
        spx = yf.download("^GSPC", period="6mo", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
        if spx.empty or len(spx) < 20:
            macro["spx_trend"], macro["spx_5d_vs_20d"] = None, None
        else:
            ma5_last  = float(spx.rolling(5).mean().iloc[-1])
            ma20_last = float(spx.rolling(20).mean().iloc[-1])
            macro["spx_trend"] = "Bullish" if ma5_last > ma20_last else "Bearish"
            macro["spx_5d_vs_20d"] = round(((ma5_last - ma20_last) / (ma20_last if ma20_last else 1)) * 100, 2)
    except Exception:
        macro["spx_trend"], macro["spx_5d_vs_20d"] = None, None

    # --- FRED CSV helpers
    def fred_csv_last(series_id: str) -> pd.DataFrame:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.rename(columns={series_id: "value"}).dropna()
        df = df[df["value"] != "."]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna()

    # --- CPI YoY from CPIAUCSL (index)
    try:
        cpi_df = fred_csv_last("CPIAUCSL").sort_values("DATE")
        last = cpi_df.iloc[-1]["value"]
        prev12 = cpi_df.iloc[-13]["value"] if len(cpi_df) > 13 else np.nan
        macro["cpi_yoy"] = round(((last/prev12) - 1) * 100, 2) if prev12 == prev12 else None
        macro["cpi_date"] = cpi_df.iloc[-1]["DATE"].date().isoformat()
    except Exception:
        macro["cpi_yoy"], macro["cpi_date"] = None, None

    # --- Unemployment rate
    try:
        un_df = fred_csv_last("UNRATE").sort_values("DATE")
        macro["unemp_rate"] = round(float(un_df.iloc[-1]["value"]), 2)
        macro["unemp_date"] = un_df.iloc[-1]["DATE"].date().isoformat()
    except Exception:
        macro["unemp_rate"], macro["unemp_date"] = None, None

    return macro

# =========================
# Indicators (1-D safe)
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float), df["Volume"].astype(float)

    # MAs / EMA
    out["MA20"]  = c.rolling(20, min_periods=1).mean()
    out["MA50"]  = c.rolling(50, min_periods=1).mean()
    out["MA200"] = c.rolling(200, min_periods=1).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()

    # RSI (Wilder)
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    out["RSI"] = (100 - (100/(1+rs))).fillna(50)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"]   = out["MACD"] - out["MACD_Signal"]

    # Bollinger Bands
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std(ddof=0)
    out["BB_Up"]  = bb_mid + 2*bb_std
    out["BB_Low"] = bb_mid - 2*bb_std
    # Optional width (1-D safe)
    out["BB_Width"] = ((out["BB_Up"] - out["BB_Low"]) / c.replace(0, np.nan)).fillna(0)

    # ATR
    prev_close = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=1).mean()

    # ADX (safe flatten)
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

    # Volume spike flag
    vol_ma = v.rolling(20, min_periods=1).mean()
    out["Vol_Spike"] = (v > 2*vol_ma).astype(int)

    out["Close"] = c
    return out.bfill().ffill()

# =========================
# News + Sentiment (API or RSS fallback)
# =========================
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

# =========================
# Analyst Pulse (best-effort)
# =========================
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
        ups   = actions.str.contains("upgrade").sum()
        downs = actions.str.contains("downgrade").sum()
        total = ups + downs
        buy_ratio = (ups / total) if total > 0 else None
        return {"buy_ratio": buy_ratio, "samples": int(total)}
    except Exception:
        return {"buy_ratio": None, "samples": 0}

def market_confidence(sentiment: float, buy_ratio: float | None):
    sent_norm = (sentiment + 1) / 2   # -1..1 -> 0..1
    if buy_ratio is None:
        conf = 0.6 * sent_norm + 0.4 * 0.5
        label = "Based on sentiment"
    else:
        conf = 0.6 * sent_norm + 0.4 * buy_ratio
        label = "Sentiment + analyst pulse"
    return int(round(conf * 100)), label

# =========================
# Signal logic + explain
# =========================
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

    # Sentiment
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

# =========================
# Backtest preview (simple)
# =========================
def backtest_preview(df: pd.DataFrame, ind: pd.DataFrame) -> float:
    sig = (ind["RSI"] < 35).astype(int) - (ind["RSI"] > 65).astype(int)
    nxt = df["Close"].shift(-5)
    ret = (nxt - df["Close"]) / df["Close"]
    sig = np.ravel(sig.values); ret = np.ravel(ret.values)
    mask = sig != 0
    if mask.sum() < 10: return 0.0
    acc = np.mean(np.sign(ret[mask]) == sig[mask])
    return round(acc * 100, 1)

# =========================
# Adaptive DCA simulator
# =========================
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
        bb_low = float(ind.loc[dt, "BB_Low"])
        atr   = float(ind.loc[dt, "ATR"])

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
                trades.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "side": "BUY",
                    "price": round(price, 2),
                    "invested": round(invest, 2),
                    "shares": round(buy_shares, 6)
                })

        # Partial take-profit vs dynamic target
        target_price = float(ind["Close"].iloc[-1] + 2*ind["ATR"].iloc[-1])
        if shares > 0 and price >= target_price:
            sell_shares = shares * 0.20
            proceeds = sell_shares * price
            shares -= sell_shares
            cash   += proceeds
            trades.append({
                "date": dt.strftime("%Y-%m-%d"),
                "side": "SELL",
                "price": round(price, 2),
                "invested": -round(proceeds, 2),
                "shares": -round(sell_shares, 6)
            })

        equity = float(shares * price + cash)
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        dd_pct = (equity - peak_equity) / (peak_equity if peak_equity else 1)
        if dd_pct < -0.30:
            halt_buys = True

    final_value = shares * df["Close"].iloc[-1] + cash
    total_invested = cash_start - cash if cash_start >= cash else sum(
        max(0, t.get("invested", 0)) for t in [{"invested": x.get("invested", 0)} for x in trades]
    )
    pnl = final_value - total_invested
    roi_pct = (pnl / total_invested * 100) if total_invested > 0 else 0.0

    ec = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(ec) if ec.size else np.array([0])
    dd = (ec - running_max) / np.where(running_max == 0, 1, running_max)
    max_dd = float(np.min(dd)) if dd.size else 0.0

    trades_df = pd.DataFrame(trades)
    return dict(
        final_value=float(final_value),
        total_invested=float(total_invested),
        roi_pct=float(roi_pct),
        max_drawdown_pct=round(100*max_dd, 2),
        trades=trades_df
    )

# =========================
# PLOTS
# =========================
def plot_dashboard(ind: pd.DataFrame, ticker: str, show_zones=True):
    last = ind.iloc[-1]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05, subplot_titles=("Price / MAs / Bollinger + Zones", "MACD", "RSI")
    )
    # Price
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

# =========================
# NEW: Feature builder
# =========================
def build_features(df: pd.DataFrame,
                   ind: pd.DataFrame,
                   macro: dict | None = None,
                   pulse: dict | None = None,
                   news_sent: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=ind.index)
    c = ind["Close"].astype(float)
    # returns
    X["ret1"]  = c.pct_change(1)
    X["ret5"]  = c.pct_change(5)
    X["ret10"] = c.pct_change(10)
    # techs
    for col in ["RSI","MACD","MACD_Signal","MACD_Hist","MA20","MA50","MA200","ADX","ATR","BB_Up","BB_Low","BB_Width","Vol_Spike"]:
        if col in ind.columns:
            X[col] = ind[col].astype(float)
    # volume deviation
    if "Volume" in df.columns:
        v = df["Volume"].astype(float)
        vma = v.rolling(20, min_periods=1).mean()
        X["Vol_dev20"] = ((v - vma) / (vma.replace(0,np.nan))).fillna(0)

    # macro constants
    if macro:
        X["SPX_5v20"] = float(macro.get("spx_5d_vs_20d")) if macro.get("spx_5d_vs_20d") is not None else 0.0
        X["VIX"]      = float(macro.get("vix_last")) if macro.get("vix_last") is not None else 0.0
        X["CPI_YoY"]  = float(macro.get("cpi_yoy")) if macro.get("cpi_yoy") is not None else 0.0
        X["Unemp"]    = float(macro.get("unemp_rate")) if macro.get("unemp_rate") is not None else 0.0
    else:
        X["SPX_5v20"] = 0.0; X["VIX"] = 0.0; X["CPI_YoY"]=0.0; X["Unemp"]=0.0

    br = (pulse or {}).get("buy_ratio", None)
    X["AnalystBuyRatio"] = float(br) if br is not None else 0.5

    X["NewsSent"] = float(news_sent)
    y = (c.shift(-5) - c) / c
    return X.fillna(0), y

# =========================
# NEW: Markov chain probs
# =========================
def markov_chain_probs(df: pd.DataFrame, horizon_days: int = 1) -> dict:
    c = df["Close"].astype(float)
    r = c.pct_change(horizon_days).dropna()
    s = np.sign(r.values)
    s[s == 0] = 1
    if len(s) < 30:
        return {"P_up": 0.5, "P_down": 0.5, "T": np.array([[0.5,0.5],[0.5,0.5]])}
    up_up = up_down = down_up = down_down = 0
    for i in range(len(s)-1):
        a, b = s[i], s[i+1]
        if a == 1 and b == 1: up_up += 1
        if a == 1 and b == -1: up_down += 1
        if a == -1 and b == 1: down_up += 1
        if a == -1 and b == -1: down_down += 1
    def norm(a,b):
        t=a+b
        return (a/t if t>0 else 0.5, b/t if t>0 else 0.5)
    p_uu, p_ud = norm(up_up, up_down)
    p_du, p_dd = norm(down_up, down_down)
    T = np.array([[p_uu, p_ud],[p_du, p_dd]])
    last_state = s[-1]
    P_up = p_uu if last_state == 1 else p_du
    return {"P_up": float(P_up), "P_down": float(1-P_up), "T": T}

# =========================
# NEW: AI forecast block
# =========================
def ai_forecast_block(df: pd.DataFrame,
                      ind: pd.DataFrame,
                      X: pd.DataFrame,
                      y: pd.Series,
                      horizon_days: int = 5) -> dict:
    X, y = X.align(y, join="inner", axis=0)
    if len(X) < 200:
        return {"pred_pct": 0.0, "conf": 0.5, "band": (0.0, 0.0)}
    split = max(150, int(len(X) * 0.8))
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:], y.iloc[split:]
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    if len(Xte) >= 20:
        pred_val = model.predict(Xte)
        resid = yte.values - pred_val
        sigma = float(np.std(resid))
    else:
        sigma = 0.02
    x_last = X.iloc[[-1]]
    pred_pct = float(model.predict(x_last)[0])
    conf = float(np.clip(abs(pred_pct) / (sigma + 1e-6), 0, 2) / 2)  # 0..1
    last_price = float(ind["Close"].iloc[-1])
    mean_target = last_price * (1 + pred_pct)
    lo = mean_target * (1 - sigma)
    hi = mean_target * (1 + sigma)
    return {"pred_pct": pred_pct, "conf": conf, "band": (lo, hi)}

# =========================
# NEW: Expected return preview
# =========================
def expected_return_preview(df: pd.DataFrame, ind: pd.DataFrame, probs: dict) -> float:
    c = ind["Close"].astype(float)
    fwd = (c.shift(-5) - c) / c
    pos = fwd[fwd > 0].mean()
    neg = -fwd[fwd < 0].mean()
    pos = float(pos) if pos == pos else 0.02
    neg = float(neg) if neg == neg else 0.02
    P_up = probs.get("P_up", 0.5)
    P_down = 1 - P_up
    er = P_up * pos - P_down * neg
    return float(er)

# =========================
# MAIN FLOW
# =========================
df = fetch_prices(ticker, horizon)
if df is None:
    st.error("No data found. Try another symbol.")
    st.stop()

ind = compute_indicators(df)
macro = fetch_macro()
headlines, news_sent = fetch_news_and_sentiment(ticker)
pulse = analyst_pulse(ticker)

decision, color, score = generate_signal(ind, news_sent, horizon)
conf_pct, conf_label = market_confidence(news_sent, pulse["buy_ratio"])

# Build features + AI forecast
X, y = build_features(df, ind, macro, pulse, news_sent)
mc = markov_chain_probs(df, horizon_days=1)
ai = ai_forecast_block(df, ind, X, y, horizon_days=5)
er = expected_return_preview(df, ind, mc)

# =========================
# Macro panel
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("VIX (volatility)", f"{macro['vix_last']:.2f}" if macro["vix_last"] is not None else "—")
m2.metric("S&P 5d vs 20d", f"{macro['spx_5d_vs_20d']:+.2f}%" if macro["spx_5d_vs_20d"] is not None else "—", macro["spx_trend"] or "")
m3.metric("CPI YoY", f"{macro['cpi_yoy']:.2f}%" if macro["cpi_yoy"] is not None else "—")
m4.metric("Unemployment", f"{macro['unemp_rate']:.2f}%" if macro["unemp_rate"] is not None else "—")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📌 Signal", "🔮 Forecast AI", "💵 Simulator", "📘 Learn"])

with tab1:
    last = ind.iloc[-1]
    cA, cB, cC, cD, cE, cF = st.columns(6)
    cA.metric("Price", f"${last['Close']:.2f}")
    cB.metric("RSI (14)", f"{last['RSI']:.1f}")
    cC.metric("MACD", f"{last['MACD']:.2f}")
    cD.metric("ADX", f"{last['ADX']:.1f}")
    cE.metric("ATR (14)", f"{last['ATR']:.2f}")
    cF.metric("Analyst Pulse", f"{int(pulse['buy_ratio']*100)}% buys" if pulse["buy_ratio"] is not None else "—")

    st.markdown(f"### **Signal: {decision}** (Score {score:+.2f}, News {news_sent:+.2f})")
    st.progress(conf_pct/100.0, text=f"Market Confidence {conf_pct}% — {conf_label}")
    st.plotly_chart(plot_dashboard(ind, ticker, show_zones=True), use_container_width=True)
    acc = backtest_preview(df, ind)
    st.write(f"**Backtest Preview (RSI swing, 5-day):** {acc:.1f}%")
    st.markdown(explain_signal(ind, news_sent, decision))
    st.progress(confidence_from_score(score))

with tab2:
    pred_pct = ai["pred_pct"] * 100
    conf = ai["conf"]
    lo, hi = ai["band"]
    last_price = float(ind["Close"].iloc[-1])

    st.subheader("AI Forecast (5-day)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Move", f"{pred_pct:+.2f}%")
    m2.metric("AI Confidence", f"{int(conf*100)}%")
    m3.metric("Price Now", f"${last_price:.2f}")
    st.caption(f"Expected range in 5d: **${lo:.2f} – ${hi:.2f}**")

    st.write(f"**Markov P(up next day):** {mc['P_up']*100:.1f}%")
    st.write(f"**Expected Return (5d, MC blend):** {er*100:.2f}%")

with tab3:
    st.subheader("Adaptive DCA (long-only) with partial take-profit")
    sim = adaptive_dca_simulator(df, ind, invest_amount)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Final Value", f"${sim['final_value']:.2f}")
    s2.metric("Total Invested", f"${sim['total_invested']:.2f}")
    s3.metric("ROI", f"{sim['roi_pct']:.1f}%")
    s4.metric("Max Drawdown", f"{sim['max_drawdown_pct']:.1f}%")
    st.markdown("#### Trades")
    if sim["trades"].empty:
        st.info("No trades executed by the rules in this period.")
    else:
        st.dataframe(sim["trades"], use_container_width=True)

    with st.expander("How this simulator works"):
        st.markdown("""
- Invests **more** when **RSI is deeply oversold** or price dips below lower Bollinger.  
- Allows **momentum entries** when MACD>Signal and MA20>MA50.  
- **Partial take-profit** when price reaches a dynamic ATR-based target.  
- Halts new buys after a **30% drawdown** from equity peak (risk control).
        """)

with tab4:
    st.markdown("""
### What you’re seeing
- **Signal Tab** uses trend (MAs, ADX), momentum (RSI, MACD), extremes (Bollinger), volume spikes and **news sentiment**.
- **Forecast AI Tab** blends a **RandomForest** (learned from historical features: returns, RSI/MACD/ATR/BB, macro, news, analyst) **plus a 2-state Markov chain** for short-term continuation.
- **Simulator Tab** demonstrates **Adaptive Dollar-Cost Averaging** (DCA) + **partial take-profit** rules.

### Educational notes
- **RSI**: <30 oversold, >70 overbought.  
- **MACD**: momentum/trend; crossovers can signal shifts.  
- **Bollinger Bands**: ±2σ from 20D mean; outside bands often mark extremes.  
- **ADX**: trend strength (>25 strong).  
- **ATR**: volatility; useful for dynamic stops/targets.  
- **Markov Chain**: estimates probability that tomorrow is up vs down given today's move.  
- **RandomForest**: non-linear ML that can combine many features without strict assumptions.

**Disclaimer:** Educational use only — this is **not** financial advice.
    """)

# =========================
# Headlines (bottom)
# =========================
with st.expander("🗞️ Latest Headlines"):
    if not headlines:
        st.write("No headlines available.")
    else:
        for h in headlines:
            title = h["title"]; url = h["url"]
            src = h.get("source",""); pub = h.get("published","")
            nice = pub[:10] if pub else ""
            st.markdown(f"- [{title}]({url}) — *{src}* {('• '+nice) if nice else ''}")
