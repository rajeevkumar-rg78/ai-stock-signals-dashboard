import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, feedparser
from io import StringIO

# ============= Banner =============
def render_header(decision: str = "HOLD"):
    decision = (decision or "").upper()
    if "BUY" in decision:
        grad = "linear-gradient(270deg, #43e97b 0%, #38f9d7 100%)"
        accent_emoji = "🟢"
    elif "SELL" in decision:
        grad = "linear-gradient(270deg, #fa709a 0%, #fee140 100%)"
        accent_emoji = "🔴"
    else:
        grad = "linear-gradient(270deg, #30cfd0 0%, #330867 100%)"
        accent_emoji = "🟠"
    st.markdown(
        f"""
        <div style="
            position:relative;
            background: {grad};
            background-size: 400% 400%;
            animation: bannerShift 8s ease-in-out infinite;
            padding: 22px 32px 44px 32px;
            border-radius: 16px;
            color: white;
            box-shadow: 0 4px 16px rgba(0,0,0,0.13);
            margin-bottom: 22px;
            overflow: hidden;
        ">
            <svg width="100%" height="40" viewBox="0 0 800 40" fill="none" xmlns="http://www.w3.org/2000/svg"
                 style="position:absolute;bottom:0;left:0;z-index:0;">
                <path d="M0 20 Q 200 60 400 20 T 800 20 V40 H0Z"
                      fill="rgba(255,255,255,0.13)" />
            </svg>
            <div style="position:relative;z-index:1;">
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div style="display:flex;align-items:center;gap:18px;">
                        <span style="font-size:38px;">🧠</span>
                        <div>
                            <div style="font-size:25px;font-weight:800;letter-spacing:0.3px;">
                                AI Stock Signals PRO
                            </div>
                            <div style="font-size:14.5px;opacity:0.93;">
                                Technicals • Macro • News • Analyst • AI Forecast
                            </div>
                        </div>
                    </div>
                    <div style="font-size:14px;text-align:right;opacity:0.93;">
                        <b>© 2025 MarketMinds LLC</b><br>
                        <span style="font-size:12.5px;opacity:0.88;">{accent_emoji} Smarter Investing</span>
                    </div>
                </div>
            </div>
        </div>
        <style>
        @keyframes bannerShift {{
          0% {{background-position: 0% 50%;}}
          50% {{background-position: 100% 50%;}}
          100% {{background-position: 0% 50%;}}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="AI Stock Signals — PRO v5.5.3", layout="wide")
render_header("HOLD")

# ============= Ticker Input =============
ticker = st.text_input("Ticker", "", placeholder="Enter a stock symbol (e.g., MSFT)").upper().strip()
if not ticker:
    st.markdown("""
        ### 👋 Welcome to AI Stock Signals PRO!
        - Please enter a stock symbol above to get started.
        - Example: `AAPL`, `MSFT`, `NVDA`, `TSLA`, etc.
    """)
    st.stop()

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

# ============= Inputs =============
c1, c2 = st.columns([2,2])
with c1:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c2:
    invest_amount = st.slider("Simulation amount ($)", min_value=500, max_value=50_000, step=500, value=10_000)

# ============= Data Fetchers (add your existing fetchers here) =============
# ... (fetch_prices_tf, fetch_earnings_date, fetch_major_indices, fetch_fundamentals, fetch_macro, etc.) ...

# ============= Fetch Data for User's Ticker =============
df = fetch_prices_tf(ticker, period, interval)
if df is None or df.empty:
    st.error(f"No data found for {ticker}.")
    st.stop()
ind = compute_indicators(df)

# ============= Price Display =============
last = ind.iloc[-1]
prev = ind.iloc[-2] if len(ind) > 1 else last
price = last["Close"]
change = price - prev["Close"]
change_pct = (change / prev["Close"]) * 100 if prev["Close"] != 0 else 0
st.metric("Price", f"${price:.2f}", delta=f"{change:+.2f} ({change_pct:+.2f}%)")

# ============= Analyst Pulse =============
pulse = analyst_pulse(ticker)
buy = pulse.get("buy") or 0
hold = pulse.get("hold") or 0
sell = pulse.get("sell") or 0
if buy > hold and buy > sell:
    accent = "#28a745"
    mood = "Bullish"
elif sell > buy and sell > hold:
    accent = "#dc3545"
    mood = "Bearish"
else:
    accent = "#f0ad4e"
    mood = "Neutral"
st.markdown(
    f"### 🧭 <span style='color:{accent};'>Analyst Pulse — {mood}</span>",
    unsafe_allow_html=True,
)
render_analyst_pulse(pulse)

# ============= Signal =============
headlines, news_sent = fetch_news_and_sentiment(ticker)
decision, color, score = generate_signal(ind, news_sent, horizon)
conf_overall = market_confidence(news_sent, pulse["buy"])
sig_conf = confidence_from_score(score)
st.markdown(f"### **Signal: {decision}** (Score {score:+.2f}, News {news_sent:+.2f})")
st.progress(conf_overall, text=f"Market Confidence {int(conf_overall*100)}% — sentiment/analyst blend")
st.metric("Signal Strength", f"{int(sig_conf*100)}%", delta=f"{score:+.2f}")
target_up   = last["Close"] + 2.0*last["ATR"]
buy_zone    = last["Close"] - 1.5*last["ATR"]
stop_loss   = last["Close"] - 2.5*last["ATR"]
st.write(f"📈 **Target (≈5d):** ${target_up:.2f} 🟦 **Buy zone:** ${buy_zone:.2f} 🛑 **Stop:** ${stop_loss:.2f}")

# ============= Macro Indicators =============
macro = fetch_macro()
m1, m2, m3, m4 = st.columns(4)
m1.metric("VIX (volatility)", f"{macro['vix_last']:.2f}" if macro["vix_last"] is not None else "—")
m2.metric("S&P 5d vs 20d", f"{macro['spx_5d_vs_20d']:+.2f}%" if macro["spx_5d_vs_20d"] is not None else "—",
          macro["spx_trend"] or "")
m3.metric("CPI YoY", f"{macro['cpi_yoy']:.2f}%")
m4.metric("Unemployment", f"{macro['unemp_rate']:.2f}%")

# ============= Earnings Date =============
earnings_date = fetch_earnings_date(ticker)
st.metric("Next Earnings Date", earnings_date if earnings_date else "Not available")

# ============= Stock Fundamentals =============
fund = fetch_fundamentals(ticker)
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

# ============= Chart =============
st.plotly_chart(plot_dashboard(ind, ticker, zones=True), use_container_width=True)

# ...rest of your dashboard (DCA, AI forecast, logbook, screener, etc.)...


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

st.markdown(f"## 🔮 Future DCA Monte Carlo Simulator ({tf} interval)")

future_periods = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "YTD": 180,
    "1Y": 252,
    "2Y": 504,
    "5Y": 1260,
    "10Y": 2520,
    "ALL": 252
}
days = future_periods.get(tf, 21)
if df is None or df.empty or "Close" not in df.columns:
    st.info("Not enough data for future DCA simulation. Try a longer chart interval.")
else:
    returns_series = df["Close"].pct_change().dropna()
    # Ensure 1D
    if isinstance(returns_series, pd.DataFrame):
        # If for some reason it's a DataFrame, take the first column
        returns_series = returns_series.iloc[:, 0]
    elif isinstance(returns_series, np.ndarray) and returns_series.ndim > 1:
        # If it's a 2D array, flatten it
        returns_series = pd.Series(returns_series.flatten())
    elif not isinstance(returns_series, pd.Series):
        # If it's not a Series, try to convert
        returns_series = pd.Series(returns_series)
    returns = pd.to_numeric(returns_series, errors="coerce").values
    returns = returns[~np.isnan(returns)]
    min_required = max(10, days // 3)
    if len(returns) < min_required:
        st.info(f"Not enough data for future DCA simulation. "
                f"Need at least {min_required} daily returns, but only have {len(returns)}. "
                "Try a longer chart interval.")
    else:
        def simulate_future_prices(df, days=10, n_sims=1000):
            returns_series = df["Close"].pct_change().dropna()
            if isinstance(returns_series, pd.DataFrame):
                returns_series = returns_series.iloc[:, 0]
            elif isinstance(returns_series, np.ndarray) and returns_series.ndim > 1:
                returns_series = pd.Series(returns_series.flatten())
            elif not isinstance(returns_series, pd.Series):
                returns_series = pd.Series(returns_series)
            returns = pd.to_numeric(returns_series, errors="coerce").values
            returns = returns[~np.isnan(returns)]
            last_price = float(df["Close"].iloc[-1])
            sims = []
            for _ in range(n_sims):
                sampled_returns = np.random.choice(returns, size=days, replace=True)
                prices = [last_price]
                for r in sampled_returns:
                    prices.append(prices[-1] * (1 + r))
                sims.append(prices[1:])
            return np.array(sims)

        def dca_on_simulated_paths(sim_prices, invest_amount, dca_freq=1):
            n_sims, n_days = sim_prices.shape
            results = []
            for sim in sim_prices:
                cash = invest_amount
                shares = 0
                for i in range(0, n_days, dca_freq):
                    price = sim[i]
                    buy_amt = cash / ((n_days - i) // dca_freq + 1)
                    shares += buy_amt / price
                    cash -= buy_amt
                final_value = shares * sim[-1] + cash
                results.append(final_value)
            return np.array(results)

        sim_prices = simulate_future_prices(df, days=days, n_sims=1000)
        
        if sim_prices is not None:
            dca_results = dca_on_simulated_paths(sim_prices, invest_amount)
            st.markdown(f"**Simulated DCA outcome for {tf} ({days} trading days):**")
            st.write(f"Mean: ${np.mean(dca_results):,.2f}")
            st.write(f"Median: ${np.median(dca_results):,.2f}")
            st.write(f"5th percentile: ${np.percentile(dca_results, 5):,.2f}")
            st.write(f"95th percentile: ${np.percentile(dca_results, 95):,.2f}")
        
             # --- Predicted future share price from Monte Carlo simulation ---

        if sim_prices is not None:
            predicted_prices = sim_prices[:, -1]
            mean_price = np.mean(predicted_prices)
            median_price = np.median(predicted_prices)
            low_price = np.percentile(predicted_prices, 2.5)
            high_price = np.percentile(predicted_prices, 97.5)
            buy_price = float(df["Close"].iloc[-1])  # Ensure this is a float, not a Series
            expected_gain = mean_price - buy_price
            expected_gain_pct = (expected_gain / buy_price) * 100 if buy_price != 0 else 0
            
            st.markdown(f"### 📈 Predicted Share Price in {days} Days ({tf})")
            st.write(f"**Current price:** ${buy_price:.2f}")
            st.write(f"**Predicted mean price:** ${mean_price:.2f}")
            st.write(f"**Median price:** ${median_price:.2f}")
            st.write(f"**95% confidence range:** ${low_price:.2f} — ${high_price:.2f}")
            st.write(f"**Expected gain/loss per share:** ${expected_gain:+.2f} ({expected_gain_pct:+.2f}%)")

            
                        


           
            # Optional: Show a line chart of the sorted predicted prices
            
            #sorted_prices = np.sort(predicted_prices)
            #st.line_chart(sorted_prices)
            
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 2.5))  # You can adjust these numbers for size
            ax.hist(dca_results, bins=30, color="#1976d2", alpha=0.7)
            ax.set_title(f"Future DCA Portfolio Value Distribution ({tf})", fontsize=12)
            ax.set_xlabel("Portfolio Value ($)", fontsize=10)
            ax.set_ylabel("Simulations", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        

            import plotly.graph_objects as go

            sorted_results = np.sort(dca_results)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=sorted_results,
                mode='lines',
                line=dict(color="#1976d2"),
                name="Simulated DCA Outcomes"
            ))
            
            
            #fig.update_layout(
                #title=f"Future DCA Portfolio Value Distribution ({tf})",
                #xaxis_title="Simulation # (sorted)",
                #yaxis_title="Portfolio Value ($)",
                #template="plotly_white"
           # )
            #st.plotly_chart(fig, use_container_width=True)
            

# Headlines
with st.expander("🗞️ Latest Headlines"):
    if not headlines:
        st.write("No headlines available.")
    else:
        for h in headlines:
            title = h["title"]; url = h["url"]; src = h.get("source",""); pub = h.get("published","")
            nice = pub[:10] if pub else ""
            st.markdown(f"- [{title}]({url}) — *{src}* {('• '+nice) if nice else ''}")


# ... your main ticker analysis code ...
# (chart, metrics, daily action, DCA, etc.)

# --- Insert the Tech Stock Recommendation Block Here ---

import pandas as pd

st.markdown("## 📋 Today's Top Tech Stock Buy Candidates")

tech_watchlist = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "TSLA", "CRM",
    "ADBE", "ORCL", "INTC", "AMD", "CSCO", "NFLX", "AVGO", "QCOM", "IBM", "SHOP", "UBER"
]

buy_candidates = []

for scan_ticker in tech_watchlist:
    try:
        scan_df = fetch_prices_tf(scan_ticker, period, interval)
        if scan_df is None or len(scan_df) < 30:
            continue
        scan_ind = compute_indicators(scan_df)
        headlines, news_sent = fetch_news_and_sentiment(scan_ticker)
        signal, color, score = generate_signal(scan_ind, news_sent, horizon)
        earnings_date = fetch_earnings_date(scan_ticker)
        last = scan_ind.iloc[-1]
        price = last["Close"]
        buy_zone = price - 1.5 * last["ATR"]
        target_up = price + 2.0 * last["ATR"]
        stop_loss = price - 2.5 * last["ATR"]

        if signal == "BUY" and price <= buy_zone * 1.05:
            buy_candidates.append({
                "Ticker": scan_ticker,
                "Price": f"${price:.2f}",
                "Score": score,
                "Buy Zone": f"${buy_zone:.2f}",
                "Target": f"${target_up:.2f}",
                "Stop": f"${stop_loss:.2f}",
                "Earnings": earnings_date,
                "News Sentiment": f"{news_sent:+.2f}"
            })
    except Exception as e:
        st.write(f"Error processing {scan_ticker}: {e}")

# Fetch data for the user's selected ticker (again, after the screener)
df = fetch_prices_tf(ticker, period, interval)
if df is None or df.empty:
    st.error(f"No data found for {ticker}.")
    st.stop()
ind = compute_indicators(df)

# Sort by score (strongest first)
buy_candidates = sorted(buy_candidates, key=lambda x: x["Score"], reverse=True)

if buy_candidates:
    st.dataframe(pd.DataFrame(buy_candidates))
else:
    st.info("No strong tech stock buy candidates found today based on your criteria.")

# ... rest of your dashboard (headlines, learn, disclaimer, etc.) ...

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
© 2025 <b>MarketMinds LLC</b> — <i>AI Stock Signals PRO </i>
</div>
    """,
    unsafe_allow_html=True,
)

