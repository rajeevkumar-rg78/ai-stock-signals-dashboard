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

st.set_page_config(page_title="AI Stock Signals — PRO ", layout="wide")
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

# ============= Data Fetchers =============
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
        if isinstance(cal, dict) and "Earnings Date" in cal:
            val = cal["Earnings Date"]
            if isinstance(val, (list, np.ndarray)):
                val = val[0]
            if hasattr(val, "strftime"):
                return val.strftime("%Y-%m-%d")
            return str(val)
        return None
    except Exception:
        return None

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

# ... (add your other fetchers: fetch_macro, compute_indicators, fetch_news_and_sentiment, analyst_pulse, generate_signal, confidence_from_score, plot_dashboard, etc.) ...

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
    fcols[0].metric("Open", fund.get('Open'))
    fcols[1].metric("High", fund.get('High'))
    fcols[2].metric("Low", fund.get('Low'))
    fcols[3].metric("Volume", fund.get('Volume'))
    fcols[4].metric("P/E", fund.get('P/E'))
    fcols[5].metric("Market Cap", fund.get('Market Cap'))
    fcols[6].metric("52w High", fund.get('52w High'))
    fcols[7].metric("52w Low", fund.get('52w Low'))
    fcols[8].metric("Avg Vol", fund.get('Avg Vol'))
    fcols[9].metric("Yield", fund.get('Yield'))
    fcols[10].metric("Beta", fund.get('Beta'))
    fcols[11].metric("EPS", fund.get('EPS'))

# ============= Chart =============
st.plotly_chart(plot_dashboard(ind, ticker, zones=True), use_container_width=True)

# ============= DCA, AI forecast, logbook, screener, etc. =============
# ... (add your DCA, AI forecast, logbook, screener, and other sections here) ...

# ============= Footer =============
st.markdown(
    """
<div style='text-align:left; color:gray; font-size:14px; line-height:1.5; margin-top:10px;'>
<b>Disclaimer:</b><br>
This dashboard is for <b>educational and informational purposes only</b> and 
<b>does not constitute financial advice</b>.<br>
Markets carry risk; always do your own research or consult a licensed financial advisor before investing.<br><br>
© 2025 <b>MarketMinds LLC</b> — <i>AI Stock Signals PRO</i>
</div>
    """,
    unsafe_allow_html=True,
)
