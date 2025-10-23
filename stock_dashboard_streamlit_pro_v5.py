# stock_dashboard_streamlit_pro_v5.py
# ---------------------------------------------------------------
# AI Stock Signals — PRO v5.0
# - Explanations (Why BUY/SELL/HOLD)
# - Pattern detectors: Double Bottom, Cup&Handle (lite), Consolidation/Breakout
# - Stronger SELL logic
# - Target/Stop suggestions via ATR + pattern levels
# - Backtest + DCA simulators (from v4)
# - No external API keys required (news/fundamentals arrive in v5.1+)
# ---------------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="AI Stock Signals — PRO v5", layout="wide")
st.title("📈 AI Stock Signals — PRO v5")
st.caption("Explanations • Pattern detection • Targets & Stops • Backtests • DCA")

# ----------------------- CONTROLS ------------------------------
c1, c2, c3, c4 = st.columns([2,2,2,2])
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with c2:
    horizon = st.radio("Horizon", ["Short-Term (1–4 weeks)", "Long-Term (3–12 months)"], index=0, horizontal=True)
with c3:
    mode = st.radio("Risk", ["Aggressive", "Moderate", "Conservative"], index=1, horizontal=True)
with c4:
    lookback_years = st.selectbox("History", ["5y","10y"], index=0)

# ----------------------- FETCH PRICES --------------------------
@st.cache_data(ttl=3600)
def fetch_prices(t: str, period="5y", interval="1d") -> pd.DataFrame | None:
    for p in [period, "5y", "2y"]:
        try:
            df = yf.download(t, period=p, interval=interval, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].dropna()
                try:
                    df.index = df.index.tz_localize(None)
                except Exception:
                    pass
                return df
        except Exception:
            pass
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
    out["BB_W"]   = (out["BB_Up"] - out["BB_Low"]) / (c.replace(0,np.nan))

    # ATR
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=1).mean()

    # ADX (14) — light impl
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
    out["High"]  = h
    out["Low"]   = l
    out["Volume"] = v
    return out.bfill().ffill()

# ----------------------- PATTERN DETECTORS ---------------------
def detect_double_bottom(ind: pd.DataFrame, window=140, tolerance=0.035):
    """
    Heuristic: lookback window; find two local lows ~same price (within tolerance) with a higher peak between.
    Returns dict with detected flag and neckline/targets if found.
    """
    ser = ind["Close"].tail(window)
    if len(ser) < 60:
        return {"found": False}
    x = ser.values
    idx = ser.index

    # local minima by simple neighbor check
    mins = []
    for i in range(2, len(x)-2):
        if x[i] < x[i-1] and x[i] < x[i+1] and x[i] <= x[i-2] and x[i] <= x[i+2]:
            mins.append(i)
    if len(mins) < 2:
        return {"found": False}

    # test most recent pairs
    for j in range(len(mins)-2, -1, -1):
        i1, i2 = mins[j], mins[j+1]
        if i2 - i1 < 10 or i2 - i1 > 60:
            continue
        p1, p2 = x[i1], x[i2]
        if abs(p1 - p2) / ((p1 + p2)/2) <= tolerance:
            mid_peak = np.max(x[i1:i2+1])
            neckline = mid_peak
            last_close = x[-1]
            breakout = last_close > neckline * 1.005
            target = neckline + (neckline - min(p1, p2))  # classical target
            return {
                "found": True,
                "neckline": float(neckline),
                "lows": (float(p1), float(p2)),
                "breakout": bool(breakout),
                "target": float(target),
                "recent_dates": (str(idx[i1].date()), str(idx[i2].date()))
            }
    return {"found": False}

def detect_consolidation_breakout(ind: pd.DataFrame, look=40):
    """
    Detect squeeze: low Bollinger width percentile + breakout (close > upper band) with volume confirmation.
    """
    sub = ind.tail(look)
    if len(sub) < 25:
        return {"squeeze": False}
    bw = sub["BB_W"].dropna()
    if bw.empty:
        return {"squeeze": False}
    thresh = bw.quantile(0.25)
    is_squeeze = bw.iloc[-5:].mean() < thresh
    breakout_up = sub["Close"].iloc[-1] > sub["BB_Up"].iloc[-1] * 1.003
    vol_conf = bool(sub["Vol_Spike"].iloc[-1] == 1)
    return {
        "squeeze": bool(is_squeeze),
        "breakout_up": bool(breakout_up),
        "volume_confirm": vol_conf
    }

def detect_cup_and_handle(ind: pd.DataFrame, window=200):
    """
    Very light approximation:
    - Smooth with MA20
    - U-shape: mid low, left & right rims higher
    - handle: small pullback on right (<1/3 cup depth)
    - breakout: close > left rim
    """
    ser = ind["Close"].rolling(3, min_periods=1).mean().tail(window)
    if len(ser) < 80:
        return {"found": False}
    x = ser.values
    idx = ser.index

    # cup low approx at global min in last N
    mid = int(np.argmin(x))
    if mid < 20 or mid > len(x)-20:
        return {"found": False}
    left_rim = np.max(x[:mid])
    right_rim = np.max(x[mid:])
    if right_rim < left_rim * 0.9:  # need right rim not too weak
        return {"found": False}
    cup_depth = (left_rim - x[mid])
    if cup_depth <= 0 or cup_depth/left_rim < 0.05:
        return {"found": False}

    # crude handle: last 15 bars within 1/3 cup depth below right_rim
    last15 = x[-15:]
    if len(last15) < 10:
        return {"found": False}
    handle_ok = (right_rim - np.min(last15)) <= (cup_depth / 3.0) + 1e-6
    breakout = x[-1] > left_rim * 1.005
    target = left_rim + cup_depth  # classical
    return {
        "found": bool(handle_ok),
        "left_rim": float(left_rim),
        "right_rim": float(right_rim),
        "cup_depth": float(cup_depth),
        "breakout": bool(breakout),
        "target": float(target),
        "mid_date": str(idx[mid].date())
    }

# ----------------------- SIGNAL ENGINE v5 ----------------------
def generate_signal_v5(ind: pd.DataFrame, horizon: str, mode: str,
                       patterns: dict, market_bias: float):
    last = ind.iloc[-1]
    score = 0.0
    reasons = []

    sens = {"Aggressive": 0.7, "Moderate": 1.0, "Conservative": 1.5}[mode]

    # Adaptive RSI bands
    vol_pct = float((ind["ATR"].iloc[-1] / max(1e-9, last["Close"])) * 100)
    if "Short" in horizon:
        rsi_high = np.clip(68 + vol_pct*0.6, 64, 80)
        rsi_low  = np.clip(32 - vol_pct*0.6, 20, 36)
    else:
        rsi_high = np.clip(70 + vol_pct*0.4, 66, 82)
        rsi_low  = np.clip(30 - vol_pct*0.4, 18, 34)

    # Trend
    if last["MA50"] > last["MA200"]:
        score += (1.75 if "Long" in horizon else 0.75)
        reasons.append("MA50 > MA200 (uptrend)")
    else:
        score -= (1.25 if "Long" in horizon else 0.5)
        reasons.append("MA50 < MA200 (downtrend)")

    if last["MA20"] > last["MA50"]:
        score += (0.5 if "Long" in horizon else 1.0)
        reasons.append("MA20 > MA50 (short-term strength)")

    # ADX
    adx_thr = 24 if "Long" in horizon else 18
    if last["ADX"] > adx_thr:
        score += 0.5
        reasons.append(f"ADX {last['ADX']:.0f} > {adx_thr} (trend strength)")

    # Momentum
    if last["RSI"] < rsi_low:
        delta = 1.5 if "Short" in horizon else 0.6
        score += delta
        reasons.append(f"RSI {last['RSI']:.0f} < {rsi_low:.0f} (oversold)")
    elif last["RSI"] > rsi_high:
        delta = 1.5 if "Short" in horizon else 0.75
        score -= delta
        reasons.append(f"RSI {last['RSI']:.0f} > {rsi_high:.0f} (overbought)")

    if last["MACD"] > last["MACD_Signal"]:
        score += 1
        reasons.append("MACD > Signal (up momentum)")
    else:
        score -= 1
        reasons.append("MACD < Signal (down momentum)")

    if last["MACD_Slope"] > 0: score += 0.3
    else: score -= 0.2

    # Mean reversion (short-term)
    if "Short" in horizon:
        if last["Close"] < ind["BB_Low"].iloc[-1]:
            score += 0.6; reasons.append("Close < Lower BB (rebound setup)")
        if last["Close"] > ind["BB_Up"].iloc[-1]:
            score -= 0.6; reasons.append("Close > Upper BB (stretch)")

    # Volume
    if last["Vol_Spike"] == 1:
        score += 0.4; reasons.append("Volume spike (accumulation)")

    # Patterns
    if patterns.get("double_bottom", {}).get("found"):
        score += 1.2
        if patterns["double_bottom"]["breakout"]:
            score += 0.6; reasons.append("Double bottom breakout")
        else:
            reasons.append("Potential double bottom forming")
    if patterns.get("consolidation", {}).get("squeeze"):
        reasons.append("Squeeze (low volatility)")
        if patterns["consolidation"]["breakout_up"] and patterns["consolidation"]["volume_confirm"]:
            score += 1.0; reasons.append("Breakout + volume confirmation")
    if patterns.get("cup_handle", {}).get("found"):
        if patterns["cup_handle"]["breakout"]:
            score += 1.0; reasons.append("Cup&Handle breakout")
        else:
            reasons.append("Cup forming (watch handle)")

    # Market bias (SPY)
    score += 0.35 * market_bias
    if market_bias > 0: reasons.append("Market tailwind (SPY uptrend)")
    elif market_bias < 0: reasons.append("Market headwind (SPY downtrend)")

    # Decision thresholds
    if "Short" in horizon:
        buy_th  = 1.9 * sens * (0.95 if market_bias > 0 else 1.0)
        sell_th = -1.2 * sens * (0.95 if market_bias < 0 else 1.0)
    else:
        buy_th  = 2.3 * sens * (0.95 if market_bias > 0 else 1.0)
        sell_th = -1.4 * sens * (0.95 if market_bias < 0 else 1.0)

    if score >= buy_th:
        sig, color = "BUY", "green"
    elif score <= sell_th:
        sig, color = "SELL", "red"
    else:
        sig, color = "HOLD", "orange"

    return sig, color, round(score,2), reasons, (rsi_low, rsi_high)

# ----------------------- TARGET/STOP SUGGESTIONS ---------------
def target_stop_suggestion(ind: pd.DataFrame, signal: str, patterns: dict):
    last = ind.iloc[-1]
    atr = float(ind["ATR"].iloc[-1])
    px  = float(last["Close"])

    # Pattern targets first
    db = patterns.get("double_bottom", {})
    cup = patterns.get("cup_handle", {})

    if signal == "BUY":
        tgt = px + 1.5 * atr
        if db.get("found") and db.get("breakout"):
            tgt = max(tgt, float(db.get("target", tgt)))
        if cup.get("found") and cup.get("breakout"):
            tgt = max(tgt, float(cup.get("target", tgt)))
        stop = px - 1.0 * atr
        text = f"🎯 Target ~ {tgt:.2f} | 🛑 Stop ~ {stop:.2f} (ATR-based)"
        return tgt, stop, text
    elif signal == "SELL":
        tgt = px - 1.5 * atr
        stop = px + 1.0 * atr
        text = f"🎯 Downside target ~ {tgt:.2f} | 📍 Invalid > {stop:.2f}"
        return tgt, stop, text
    else:
        # Hold: give next key levels
        up = px + 1.0 * atr
        dn = px - 1.0 * atr
        text = f"⚖️ Range watch: {dn:.2f} — {up:.2f} (±1 ATR)"
        return up, dn, text

# ----------------------- BACKTEST (SIGNALS) --------------------
def backtest_signals(ind: pd.DataFrame, horizon: str, mode: str,
                     exit_rule: str = "Opposite", hold_days: int = 7) -> pd.DataFrame:
    df = ind.copy()
    df["Signal"] = "HOLD"
    df["Score"] = 0.0

    # market bias from its own MA50/200
    mb_series = np.where(df["MA50"] > df["MA200"], 1.0, np.where(df["MA50"] < df["MA200"], -1.0, 0.0))

    # precompute simple pattern flags on rolling basis (fast)
    for i in range(len(df)):
        if i < 220:
            continue
        sub = df.iloc[:i+1]
        pats = {
            "double_bottom": detect_double_bottom(sub),
            "consolidation": detect_consolidation_breakout(sub),
            "cup_handle": detect_cup_and_handle(sub),
        }
        sig, _, sc, _, _ = generate_signal_v5(sub, horizon, mode, pats, mb_series[i])
        df.iloc[i, df.columns.get_loc("Signal")] = sig
        df.iloc[i, df.columns.get_loc("Score")] = sc

    # Build positions
    df["Position"] = 0
    if exit_rule == "Opposite":
        pos = 0
        for i in range(1, len(df)):
            s_now = df["Signal"].iloc[i]
            if s_now == "BUY" and pos <= 0:
                pos = 1
            elif s_now == "SELL" and pos >= 0:
                pos = -1
            df.iloc[i, df.columns.get_loc("Position")] = pos
    else:
        pos, days_left = 0, 0
        for i in range(1, len(df)):
            s_now = df["Signal"].iloc[i]
            if pos == 0:
                if s_now == "BUY":  pos, days_left = 1, hold_days
                elif s_now == "SELL": pos, days_left = -1, hold_days
            else:
                days_left -= 1
                if days_left <= 0:
                    pos = 0
            df.iloc[i, df.columns.get_loc("Position")] = pos

    df["Ret"] = df["Close"].pct_change().fillna(0.0)
    df["StratRet"] = df["Ret"] * df["Position"].shift(1).fillna(0)
    df["Equity"] = (1 + df["StratRet"]).cumprod()
    df["BuyHold"] = (1 + df["Ret"]).cumprod()
    return df

# ----------------------- DCA SIMULATOR -------------------------
def dca_simulator(df: pd.DataFrame, amount=200, frequency="W"):
    alloc = df["Close"].resample(frequency).last().dropna()
    shares = (amount / alloc).fillna(0.0)
    total_shares = shares.cumsum()
    total_contrib = amount * len(alloc)
    final_value = float(total_shares.iloc[-1] * alloc.iloc[-1])
    pct = (final_value - total_contrib) / (total_contrib if total_contrib else 1)
    return float(total_contrib), float(final_value), float(pct)

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

# ----------------------- EXPLANATION TEXT ----------------------
def explain_signal(reasons: list, signal: str):
    prefix = {"BUY":"🟢 Why BUY: ", "SELL":"🔴 Why SELL: ", "HOLD":"🟠 Why HOLD: "}[signal]
    if not reasons:
        return prefix + "Mixed signals."
    # keep the 6 most important unique reasons
    uniq = []
    for r in reasons:
        if r not in uniq:
            uniq.append(r)
    return prefix + "; ".join(uniq[:6])

# ----------------------- MAIN TABS -----------------------------
tab_dash, tab_backtest, tab_sim, tab_learn = st.tabs(
    ["📊 Dashboard", "🧪 Backtest", "🧮 Simulators", "🎓 Learn"]
)

with tab_dash:
    if ticker:
        df = fetch_prices(ticker, period=lookback_years)
        if df is None:
            st.error(f"No data found for {ticker}.")
        else:
            ind = compute_indicators(df)

            # Market context (SPY bias)
            spy = fetch_index("SPY", period=lookback_years)
            market_bias = 0.0
            if spy is not None and len(spy) > 200:
                spy_ind = compute_indicators(spy)
                market_bias = 1.0 if spy_ind.iloc[-1]["MA50"] > spy_ind.iloc[-1]["MA200"] else (-1.0 if spy_ind.iloc[-1]["MA50"] < spy_ind.iloc[-1]["MA200"] else 0.0)

            # Patterns
            patterns = {
                "double_bottom": detect_double_bottom(ind),
                "consolidation": detect_consolidation_breakout(ind),
                "cup_handle": detect_cup_and_handle(ind),
            }

            # Signal
            signal, color, score, reasons, rsi_band = generate_signal_v5(ind, horizon, mode, patterns, market_bias)
            tgt, stop, text_ts = target_stop_suggestion(ind, signal, patterns)
            last = ind.iloc[-1]

            # Metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Price", f"${last['Close']:.2f}")
            m2.metric("RSI (14)", f"{last['RSI']:.1f}")
            m3.metric("MACD", f"{last['MACD']:.2f}")
            m4.metric("ADX", f"{last['ADX']:.1f}")
            m5.metric("ATR (14)", f"{last['ATR']:.2f}")
            m6.metric("BB Width", f"{(last['BB_W']*100):.2f}%")

            st.markdown(
                f"### **{signal}** (Score: {score}, Market: "
                f"{'Bull' if market_bias>0 else ('Bear' if market_bias<0 else 'Neutral')}, "
                f"Horizon: *{horizon}*)"
            )
            st.markdown(explain_signal(reasons, signal))
            st.info(text_ts)

            # Pattern badges
            badges = []
            if patterns["double_bottom"].get("found"):
                b = "✅ Double Bottom"
                if patterns["double_bottom"].get("breakout"): b += " — breakout"
                badges.append(b)
            if patterns["consolidation"].get("squeeze"):
                b = "✅ Squeeze"
                if patterns["consolidation"].get("breakout_up"): b += " — breakout↑"
                badges.append(b)
            if patterns["cup_handle"].get("found"):
                b = "✅ Cup&Handle"
                if patterns["cup_handle"].get("breakout"): b += " — breakout"
                badges.append(b)
            if badges:
                st.markdown("**Patterns:** " + " | ".join(badges))

            fig = plot_dashboard(ind, ticker)
            st.plotly_chart(fig, use_container_width=True)

with tab_backtest:
    st.subheader("Signal Backtest (Preview)")
    cA, cB, cC = st.columns([1,1,2])
    with cA:
        exit_rule = st.selectbox("Exit Rule", ["Opposite", "Time"], index=0)
    with cB:
        hold_days = st.slider("Hold Days (if Time exit)", 3, 30, 7)
    with cC:
        st.caption("Strategy equity vs Buy & Hold based on the v5 signal engine.")

    df = fetch_prices(ticker, period=lookback_years)
    if df is not None:
        ind = compute_indicators(df)
        bt = backtest_signals(ind, horizon, mode, exit_rule=exit_rule, hold_days=hold_days)
        eq = go.Figure()
        eq.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], name="Strategy", line=dict(width=2)))
        eq.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold"], name="Buy & Hold", line=dict(width=2, dash="dot")))
        eq.update_layout(title="Equity Curve (normalized to 1.0)", template="plotly_white", height=420)
        st.plotly_chart(eq, use_container_width=True)

        total_ret = bt["Equity"].iloc[-1] - 1.0
        bh_ret = bt["BuyHold"].iloc[-1] - 1.0
        daily = bt["StratRet"]
        sharpe = float(np.sqrt(252) * daily.mean() / (daily.std() + 1e-9))
        st.write(f"**Strategy:** {total_ret*100:.1f}%  |  **Buy & Hold:** {bh_ret*100:.1f}%  |  **Sharpe (approx):** {sharpe:.2f}")

with tab_sim:
    st.subheader("Simulators")

    # DCA
    st.markdown("### Dollar-Cost Averaging (DCA)")
    s1, s2, s3 = st.columns(3)
    with s1:
        dca_amt = st.number_input("Amount per period ($)", 50, 5000, 200, step=50)
    with s2:
        dca_freq = st.selectbox("Frequency", ["W (Weekly)", "M (Monthly)"], index=0)
    with s3:
        st.caption("Invest fixed amount per period and hold long term.")

    df = fetch_prices(ticker, period=lookback_years)
    if df is not None:
        freq_code = "W" if dca_freq.startswith("W") else "M"
        contrib, value, pct = dca_simulator(df, amount=dca_amt, frequency=freq_code)
        st.write(f"**Contributed:** ${contrib:,.0f}  |  **Final Value:** ${value:,.0f}  |  **Return:** {pct*100:.1f}%")

with tab_learn:
    st.subheader("Investor Education (Quick Reference)")
    with st.expander("Signals explained (what we use)"):
        st.markdown("""
- **Trend**: MA20/50/200 stacking; **MA50>MA200** favors long-term uptrend.  
- **RSI (adaptive)**: Overbought/oversold bands expand/contract with volatility (ATR).  
- **MACD & slope**: Direction + momentum acceleration.  
- **Bollinger**: Mean reversion extremes & **squeeze → breakout** timing.  
- **ADX**: Trend strength filter.  
- **Patterns**: **Double Bottom**, **Cup & Handle (lite)**, **Consolidation/Breakout**.
""")
    with st.expander("When do we BUY / SELL?"):
        st.markdown("""
**BUY** when multiple of: uptrend (MA50>MA200), RSI rebound from oversold, MACD>Signal, squeeze breakout with volume, bullish patterns (double bottom or cup breakout).  
**SELL** when: RSI overbought + weakening momentum (MACD<Signal, slope ≤ 0), upper BB stretch, trend deterioration (MA20<MA50 or MA50<MA200).
""")
    with st.expander("Targets & Stops (risk)"):
        st.markdown("""
We suggest **ATR-based** targets/stops and also use **pattern necklines** when present:  
- BUY: **Target ≈ Price + 1.5×ATR** (or pattern target), **Stop ≈ Price − 1×ATR**.  
- SELL: **Target ≈ Price − 1.5×ATR**, **Invalidation ≈ Price + 1×ATR**.
""")
    with st.expander("Backtests & Simulator"):
        st.markdown("""
**Backtest** shows strategy equity vs **Buy & Hold** using daily closes.  
**DCA** simulates fixed-amount periodic buys.  
*Educational preview only* — real trading needs costs, slippage, and out-of-sample validation.
""")

# ----------------------- DISCLAIMER ----------------------------
st.markdown("---")
st.markdown("""
**Disclaimer:** Educational use only. Not financial advice. Markets involve risk.  
© 2025 Raj Gupta — *AI Stock Signals — PRO v5.0*
""")
