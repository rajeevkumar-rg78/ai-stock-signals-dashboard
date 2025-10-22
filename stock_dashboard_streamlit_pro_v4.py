# stock_dashboard_streamlit_pro_v4.py
# ---------------------------------------------------------------
# AI Stock Signals — PRO v4
# - Real SELL signals
# - Long backtests (5–10y)
# - Signal backtest with equity curve vs Buy&Hold
# - DCA simulator (weekly/monthly) + example trade P&L
# - Optional ML: 5-day move classifier on indicators (graceful fallback)
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

# Optional ML (graceful fallback if sklearn not present)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

st.set_page_config(page_title="AI Stock Signals — PRO v4", layout="wide")
st.title("📈 AI Stock Signals — PRO v4")
st.caption("Backtests • DCA • Example P&L • Stronger SELL signals • Optional ML classifier")

# ----------------------- CONTROLS ------------------------------
top1, top2, top3, top4 = st.columns([2,2,2,2])
with top1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with top2:
    horizon = st.radio("Horizon", ["Short-Term (1–2 weeks)", "Long-Term (3–12 months)"], index=0, horizontal=True)
with top3:
    mode = st.radio("Risk", ["Aggressive", "Moderate", "Conservative"], index=1, horizontal=True)
with top4:
    use_news = st.toggle("Use Live News Sentiment", value=True)

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", None)

# ----------------------- FETCH PRICES --------------------------
@st.cache_data(ttl=3600)
def fetch_prices(t: str, period="10y", interval="1d") -> pd.DataFrame | None:
    # Try 10y, fallback to 5y if empty
    for p in [period, "5y", "2y"]:
        try:
            df = yf.download(t, period=p, interval=interval, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                try:
                    df.index = df.index.tz_localize(None)
                except Exception:
                    pass
                return df
        except Exception:
            pass
    return None

@st.cache_data(ttl=3600)
def fetch_index(symbol="SPY", period="10y", interval="1d") -> pd.DataFrame | None:
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

    # ADX (14)
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

# ----------------------- NEWS SENTIMENT ------------------------
FINANCE_DOMAINS = [
    "finance.yahoo.com","cnbc.com","bloomberg.com","marketwatch.com",
    "reuters.com","seekingalpha.com","investing.com","wsj.com","ft.com"
]

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
        txt   = f"{title}. {desc}"
        w = 1.5 if any(k in (src or "").lower() for k in ["bloomberg","cnbc","reuters","wsj","ft.com"]) else 1.0
        s = analyzer.polarity_scores(txt)["compound"]
        scores.append(w * s)
        if len(top) < 5:
            top.append((title, f"{src} | {url}"))
    return (float(np.mean(scores)) if scores else 0.0), top

def detect_events(headlines: List[str]) -> Dict[str,bool]:
    EVENT_POS = ["upgrade","raises target","initiates buy","strong buy","beats estimates","outperform","overweight","price target raised"]
    EVENT_NEG = ["downgrade","misses estimates","underperform","underweight","guidance cut","profit warning"]
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

# ----------------------- SIGNAL ENGINE v4 ----------------------
def generate_signal_v4(ind: pd.DataFrame, sentiment: float, mode: str, horizon: str,
                       events: Dict[str,bool], market_bias: float):
    last = ind.iloc[-1]
    score = 0.0

    sens = {"Aggressive": 0.7, "Moderate": 1.0, "Conservative": 1.5}[mode]

    # Adaptive RSI bands by volatility
    vol_pct = float((ind["ATR"].iloc[-1] / max(1e-9, last["Close"])) * 100)
    if "Short" in horizon:
        rsi_high = np.clip(68 + vol_pct*0.6, 64, 80)
        rsi_low  = np.clip(32 - vol_pct*0.6, 20, 36)
    else:
        rsi_high = np.clip(70 + vol_pct*0.4, 66, 82)
        rsi_low  = np.clip(30 - vol_pct*0.4, 18, 34)

    # Trend weight differs by horizon
    if "Long" in horizon:
        if last["MA50"] > last["MA200"]: score += 1.75
        else: score -= 1.25
        if last["MA20"] > last["MA50"]: score += 0.5
    else:
        if last["MA20"] > last["MA50"]: score += 1.00
        if last["MA50"] > last["MA200"]: score += 0.75

    if last["ADX"] > (24 if "Long" in horizon else 18): score += 0.5

    # Momentum (Short-term stronger)
    if "Short" in horizon:
        if last["RSI"] < rsi_low:  score += 1.5
        if last["RSI"] > rsi_high: score -= 1.5
    else:
        if last["RSI"] < rsi_low:  score += 0.5
        if last["RSI"] > rsi_high: score -= 0.75

    # MACD & slope
    score += 1 if last["MACD"] > last["MACD_Signal"] else -1
    if last["MACD_Slope"] > 0: score += 0.4
    else: score -= 0.2

    # Mean reversion (short-term timing)
    if "Short" in horizon:
        if last["Close"] < ind["BB_Low"].iloc[-1]: score += 0.75
        if last["Close"] > ind["BB_Up"].iloc[-1]:  score -= 0.75

    # Volume confirmation
    if last["Vol_Spike"]: score += 0.5

    # News + events
    score += float(np.clip(sentiment * (2.0 if "Short" in horizon else 1.4), -1.6, 1.6))
    if events.get("upgrade"):   score += (2.0 if "Short" in horizon else 1.4)
    if events.get("downgrade"): score -= (2.0 if "Short" in horizon else 1.4)

    # Market bias (SPY trend)
    score += 0.35 * market_bias

    # Thresholds — make SELL more achievable than v3.5
    if "Short" in horizon:
        buy_th  = 1.9 * sens * (0.95 if market_bias > 0 else 1.0)
        sell_th = -1.2 * sens * (0.95 if market_bias < 0 else 1.0)
    else:
        buy_th  = 2.3 * sens * (0.95 if market_bias > 0 else 1.0)
        sell_th = -1.4 * sens * (0.95 if market_bias < 0 else 1.0)

    if score >= buy_th:
        return "BUY", "green", round(score, 2), (rsi_low, rsi_high)
    elif score <= sell_th:
        return "SELL", "red", round(score, 2), (rsi_low, rsi_high)
    else:
        return "HOLD", "orange", round(score, 2), (rsi_low, rsi_high)

# ----------------------- BACKTEST (SIGNALS) --------------------
def backtest_signals(ind: pd.DataFrame,
                     horizon: str,
                     mode: str,
                     sentiment_series: pd.Series | None = None,
                     events_series: pd.Series | None = None,
                     market_bias_series: pd.Series | None = None,
                     exit_rule: str = "Opposite",
                     hold_days: int = 7) -> pd.DataFrame:
    """
    Simulates trading based on daily signals.
    exit_rule: "Opposite" or "Time"
    """
    df = ind.copy()
    df["SignalScore"] = 0.0
    df["Signal"] = "HOLD"

    # Simplified daily signal (no news series by date -> use zero)
    zero = 0.0
    ev = {"upgrade": False, "downgrade": False}

    # Market bias series -> approximate with MA50 vs MA200 of itself
    mb = 0.0
    for i in range(len(df)):
        row = df.iloc[: i+1]
        if len(row) < 210:
            df.iloc[i, df.columns.get_loc("SignalScore")] = 0.0
            df.iloc[i, df.columns.get_loc("Signal")] = "HOLD"
            continue
        ev_today = ev
        mb_today = mb
        sig, _, score, _ = generate_signal_v4(row, zero, mode, horizon, ev_today, mb_today)
        df.iloc[i, df.columns.get_loc("SignalScore")] = score
        df.iloc[i, df.columns.get_loc("Signal")] = sig

    # Create trades
    df["Position"] = 0
    if exit_rule == "Opposite":
        pos = 0
        for i in range(1, len(df)):
            s_prev, s_now = df["Signal"].iloc[i-1], df["Signal"].iloc[i]
            if s_now == "BUY" and pos <= 0:
                pos = 1
            elif s_now == "SELL" and pos >= 0:
                pos = -1
            elif s_now == "HOLD":
                pos = pos
            df.iloc[i, df.columns.get_loc("Position")] = pos
    else:
        # Time-based exit
        pos = 0
        days_left = 0
        for i in range(1, len(df)):
            s_now = df["Signal"].iloc[i]
            if pos == 0:
                if s_now == "BUY":
                    pos = 1; days_left = hold_days
                elif s_now == "SELL":
                    pos = -1; days_left = hold_days
            else:
                days_left -= 1
                if days_left <= 0:
                    pos = 0
            df.iloc[i, df.columns.get_loc("Position")] = pos

    # Compute returns
    df["CloseShift"] = df["Close"].shift(1)
    df["Ret"] = df["Close"].pct_change().fillna(0.0)
    df["StratRet"] = df["Ret"] * df["Position"].shift(1).fillna(0)
    df["Equity"] = (1 + df["StratRet"]).cumprod()
    df["BuyHold"] = (1 + df["Ret"]).cumprod()
    return df

# ----------------------- DCA SIMULATOR -------------------------
def dca_simulator(df: pd.DataFrame, amount=200, frequency="W") -> Tuple[float, float, float]:
    """
    Invest 'amount' every frequency ("W" weekly, "M" monthly) at close.
    Returns (total_contrib, final_value, pct_return).
    """
    alloc = df["Close"].resample(frequency).last().dropna()
    shares = (amount / alloc).fillna(0.0)
    total_shares = shares.cumsum()
    total_contrib = amount * len(alloc)
    final_value = float(total_shares.iloc[-1] * alloc.iloc[-1])
    pct = (final_value - total_contrib) / (total_contrib if total_contrib else 1)
    return float(total_contrib), float(final_value), float(pct)

# ----------------------- EXAMPLE TRADE P&L ---------------------
def example_trade_pnl(ind: pd.DataFrame, shares_per_buy=10, lookback_days=180, hold_days=10):
    """
    'What if I bought X shares on each BUY over last N days and sold after H days?'
    """
    sub = ind.tail(lookback_days).copy()
    sub["Entry"] = (sub["RSI"] < 35) & (sub["MACD"] > sub["MACD_Signal"])  # BUY condition proxy
    entries = sub[sub["Entry"]].index.tolist()
    trades = []
    for dt in entries:
        entry_idx = sub.index.get_loc(dt)
        exit_idx = min(entry_idx + hold_days, len(sub)-1)
        entry_price = float(sub.loc[dt, "Close"])
        exit_price = float(sub.iloc[exit_idx]["Close"])
        pnl = (exit_price - entry_price) * shares_per_buy
        trades.append({"date": dt.date(), "entry": entry_price, "exit": exit_price,
                       "shares": shares_per_buy, "pnl_$": pnl, "ret_%": (exit_price/entry_price - 1)*100})
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)

# ----------------------- OPTIONAL ML ---------------------------
def ml_predict_next5d(ind: pd.DataFrame) -> Tuple[float, str]:
    """
    Train a simple time-series logistic model to classify next-5d return > 0.
    Returns (prob_up, note). Gracefully degrades if sklearn missing or data short.
    """
    if not HAVE_SKLEARN or len(ind) < 300:
        return 0.5, ("(ML preview unavailable — install scikit-learn or lengthen data)")
    X = ind[["RSI","MACD","MACD_Signal","MACD_Hist","MA10","MA20","MA50","MA200","ADX","ATR"]].copy()
    y = (ind["Close"].shift(-5) / ind["Close"] - 1.0 > 0).astype(int)  # 1 if 5d ahead up
    X, y = X.iloc[:-5], y.iloc[:-5]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    # Fit on all for now (fast preview); for production, do CV estimate
    pipe.fit(X, y)
    prob_up = float(pipe.predict_proba(X.iloc[[-1]])[0][1])
    return prob_up, "(ML preview on indicators; prob next 5d ↑)"

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

    fig.update_layout(height=820, title=f"{ticker} — Technical Dashboard",
                      template="plotly_white", legend=dict(orientation="h", y=-0.08))
    return fig

# ----------------------- MAIN TABS -----------------------------
tab_dash, tab_backtest, tab_sim, tab_learn = st.tabs(
    ["📊 Dashboard", "🧪 Backtest", "🧮 Simulators", "🎓 Learn"]
)

with tab_dash:
    if ticker:
        df = fetch_prices(ticker)
        if df is None:
            st.error(f"No data found for {ticker}.")
        else:
            ind = compute_indicators(df)

            # Market context
            spy = fetch_index("SPY")
            market_bias = 0.0
            if spy is not None and len(spy) > 200:
                spy_ind = compute_indicators(spy)
                market_bias = 1.0 if spy_ind.iloc[-1]["MA50"] > spy_ind.iloc[-1]["MA200"] else (-1.0 if spy_ind.iloc[-1]["MA50"] < spy_ind.iloc[-1]["MA200"] else 0.0)

            # News
            sentiment, events, headlines = 0.0, {"upgrade": False, "downgrade": False}, []
            news_note = ""
            if use_news:
                if NEWS_API_KEY:
                    sentiment, events, headlines = fetch_news(ticker, NEWS_API_KEY)
                else:
                    news_note = "🔒 Add NEWS_API_KEY in Streamlit Secrets to enable live finance headlines."

            # Signal
            signal, color, score, rsi_band = generate_signal_v4(ind, sentiment, mode, horizon, events, market_bias)
            last = ind.iloc[-1]

            # Optional ML
            prob_up, ml_note = ml_predict_next5d(ind)

            # Metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Price", f"${last['Close']:.2f}")
            m2.metric("RSI (14)", f"{last['RSI']:.1f}")
            m3.metric("MACD", f"{last['MACD']:.2f}")
            m4.metric("ADX", f"{last['ADX']:.1f}")
            m5.metric("ATR (14)", f"{last['ATR']:.2f}")
            m6.metric("ML prob next 5d ↑", f"{prob_up*100:.0f}%")

            st.markdown(
                f"### **{signal}**  "
                f"(Score: {score}, Sentiment: {sentiment:+.2f}, Market: "
                f"{'Bull' if market_bias>0 else ('Bear' if market_bias<0 else 'Neutral')}, "
                f"Horizon: *{horizon}*)"
            )
            if ml_note:
                st.caption(ml_note)

            fig = plot_dashboard(ind, ticker)
            st.plotly_chart(fig, use_container_width=True)

            if use_news:
                st.markdown("#### Latest Finance Headlines")
                if headlines:
                    for title, src in headlines:
                        st.write(f"- {title}  \n  <span style='color:#888;font-size:0.9em'>{src}</span>", unsafe_allow_html=True)
                else:
                    st.info(news_note or "No recent finance headlines found.")

with tab_backtest:
    st.subheader("Signal Backtest (Preview)")
    colA, colB, colC = st.columns([1,1,2])
    with colA:
        exit_rule = st.selectbox("Exit Rule", ["Opposite", "Time"], index=0)
    with colB:
        hold_days = st.slider("Hold Days (if Time exit)", 3, 30, 7)
    with colC:
        st.caption("Compare strategy vs Buy & Hold using the daily generated signals.")

    df = fetch_prices(ticker)
    if df is not None:
        ind = compute_indicators(df)
        bt = backtest_signals(ind, horizon, mode, exit_rule=exit_rule, hold_days=hold_days)
        # Plot equity curve
        eq = go.Figure()
        eq.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], name="Strategy", line=dict(width=2)))
        eq.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold"], name="Buy & Hold", line=dict(width=2, dash="dot")))
        eq.update_layout(title="Equity Curve (normalized to 1.0)", template="plotly_white", height=420)
        st.plotly_chart(eq, use_container_width=True)

        # Metrics
        total_ret = bt["Equity"].iloc[-1] - 1.0
        bh_ret = bt["BuyHold"].iloc[-1] - 1.0
        daily = bt["StratRet"]
        sharpe = float(np.sqrt(252) * daily.mean() / (daily.std() + 1e-9))
        st.write(f"**Strategy Return:** {total_ret*100:.1f}%  |  **Buy & Hold:** {bh_ret*100:.1f}%  |  **Sharpe (approx):** {sharpe:.2f}")

with tab_sim:
    st.subheader("Simulators")

    # DCA
    st.markdown("### Dollar-Cost Averaging (DCA)")
    col1, col2, col3 = st.columns(3)
    with col1:
        dca_amt = st.number_input("Amount per period ($)", 50, 5000, 200, step=50)
    with col2:
        dca_freq = st.selectbox("Frequency", ["W (Weekly)", "M (Monthly)"], index=0)
    with col3:
        st.caption("Invest fixed amount each period and hold.")

    df = fetch_prices(ticker)
    if df is not None:
        freq_code = "W" if dca_freq.startswith("W") else "M"
        contrib, value, pct = dca_simulator(df, amount=dca_amt, frequency=freq_code)
        st.write(f"**Total Contributed:** ${contrib:,.0f}  |  **Final Value:** ${value:,.0f}  |  **Return:** {pct*100:.1f}%")

        # Example P&L
        st.markdown("### Example Trade P&L (recent BUYs)")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            shares_each = st.number_input("Shares per BUY", 1, 1000, 10, step=1)
        with tcol2:
            lookback = st.slider("Lookback Days", 60, 365*2, 180, step=30)
        with tcol3:
            holdN = st.slider("Hold Days after BUY", 3, 30, 10)

        ind = compute_indicators(df)
        trades = example_trade_pnl(ind, shares_per_buy=shares_each, lookback_days=lookback, hold_days=holdN)
        if trades.empty:
            st.info("No recent BUY-like opportunities under the example rule (RSI<35 & MACD>Signal).")
        else:
            st.dataframe(trades, use_container_width=True)
            total_pnl = trades["pnl_$"].sum()
            avg_ret = trades["ret_%"].mean()
            st.write(f"**Total P&L:** ${total_pnl:,.2f}  |  **Avg trade return:** {avg_ret:.2f}%")

with tab_learn:
    st.subheader("Investor Education (Quick Reference)")
    with st.expander("Short vs Long-Term Signals"):
        st.markdown("""
- **Short-Term:** Momentum & timing — RSI bands adapt by volatility, MACD slope, Bollinger touches, volume spikes.
- **Long-Term:** Trend & durability — MA50 vs MA200, ADX for trend strength, momentum confirmation via MACD.
""")
    with st.expander("When do we SELL?"):
        st.markdown("""
We now trigger **SELL** more readily when:
- **RSI** moves above adaptive overbought band,  
- **MACD < Signal** and **MACD_Slope ≤ 0**,  
- **Close > Upper Bollinger** (overextension),  
- **Trend weakens** (MA20 < MA50 or MA50 < MA200),  
- **Negative news** (downgrade/miss) when news is enabled.
""")
    with st.expander("Backtesting notes"):
        st.markdown("""
This is a **preview backtest** for education. It uses daily close-to-close changes and simple rules.
For production: add transaction costs, slippage, survivorship bias checks, walk-forward validation and OOS tests.
""")

# ----------------------- DISCLAIMER ----------------------------
st.markdown("---")
st.markdown("""
**Disclaimer:** Educational use only. Not financial advice. Markets involve risk.  
© 2025 Raj Gupta — *AI Stock Signals — PRO v4*
""")
