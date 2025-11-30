# app.py — AI Stock Signals PRO (Business-Ready UI)
# ----------------------------------------------------------------------
# Clean layout flow:
# Inputs → Signal Card → Chart → Why → Forecast → [Sim Tabs]
# → Macro & Fundamentals & Analyst Pulse → Tech Screener → News → Learn → Disclaimer
#
# Notes:
# - No paper-trading state/logs.
# - Rate-limit safe analyst_pulse (soft-fail).
# - Silent fallbacks on Yahoo News/FRED where appropriate.F
# - Keep your core logic intact; UI is restructured for business use.
# ----------------------------------------------------------------------

import streamlit as st
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests, feedparser, time, random
from io import StringIO
import matplotlib.pyplot as plt
import bcrypt

if "user" not in st.session_state:
    st.session_state.user = None


from supabase import create_client
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)



def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default
 
def human_fmt(val, kind=None):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if kind == "vol":
            if val >= 1e9:  return f"{val/1e9:.2f}B"
            if val >= 1e6:  return f"{val/1e6:.2f}M"
            if val >= 1e3:  return f"{val/1e3:.2f}K"
            return f"{val:.0f}"
        if kind == "cap":
            if val >= 1e12: return f"{val/1e12:.3f}T"
            if val >= 1e9:  return f"{val/1e9:.2f}B"
            if val >= 1e6:  return f"{val/1e6:.2f}M"
            return f"{val:.0f}"
        if kind == "pct":
            return f"{val*100:.2f}%"
        return f"{val:.2f}"
    except Exception:
        return "—"


# ============================================================
#  PAGE CONFIG + HEADER BANNER (FINAL CLEAN VERSION)
# ============================================================

st.set_page_config(page_title="AISigmaX — AI Stock Signals", layout="wide")

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
        <style>
        .aisigmax-banner {{
            position:relative;
            background: {grad};
            /* animation removed */
            padding: 20px 16px 36px 16px;
            border-radius: 16px;
            color: white;
            box-shadow: 0 4px 16px rgba(0,0,0,0.13);
            margin-bottom: 18px;
            overflow: hidden;
            max-width: 100vw;
        }}
        .aisigmax-flex {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}
        .aisigmax-brand {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .aisigmax-title {{
            font-size: 1.7em;
            font-family: monospace, 'Fira Mono', 'Menlo', 'Consolas', sans-serif;
            font-weight: 900;
            letter-spacing: 1.5px;
            background: linear-gradient(90deg,#1976d2,#43e97b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display:inline-block;
        }}
        .aisigmax-tagline {{
            font-size: 1em;
            opacity: 0.93;
            margin-top: 2px;
        }}
        .aisigmax-right {{
            font-size: 0.95em;
            text-align: right;
            opacity: 0.93;
            min-width: 120px;
        }}
        @media (max-width: 700px) {{
            .aisigmax-flex {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}
            .aisigmax-right {{
                text-align: left;
                margin-top: 10px;
            }}
        }}
        </style>
        <div class="aisigmax-banner">
            <svg width="100%" height="40" viewBox="0 0 800 40" fill="none" xmlns="http://www.w3.org/2000/svg"
                 style="position:absolute;bottom:0;left:0;z-index:0;">
                <path d="M0 20 Q 200 60 400 20 T 800 20 V40 H0Z" fill="rgba(255,255,255,0.13)" />
            </svg>
            <div style="position:relative;z-index:1;">
                <div class="aisigmax-flex">
                    <div class="aisigmax-brand">
                        <!-- removed funky icon span here -->
                        <div>
                            <div class="aisigmax-title">AISigmaX</div>
                            <div class="aisigmax-tagline">Next-Gen AI • Macro • News • Analyst • Forecasts</div>
                        </div>
                    </div>
                    <div class="aisigmax-right">
                        <b>&copy; 2025 MarketSignal LLC</b><br>
                        <span style="font-size:0.95em;opacity:0.88;">
                            {accent_emoji} Powered by
                            <a href="https://www.aisigmax.com"
                               style="color:white;text-decoration:underline;"
                               target="_blank">AISigmaX.com</a>
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.set_page_config(page_title="AISigmaX — AI Stock Signals", layout="wide")
render_header("HOLD")



# 🧠 Tutorial: expanded on first visit only
#if "tutorial_shown" not in st.session_state:
   #st.session_state.tutorial_shown = False

#with st.expander("🎓 Learn How AI Stock Signals Works", expanded=not st.session_state.tutorial_shown):
    #st.markdown("""
with st.expander("🎓 Learn How AI Stock Signals Works", expanded=False):
     st.markdown("""
## 🧠 What Is AISigmaX?
AI Stock Signals is an **AI-powered analytics dashboard** that blends:
- 📊 Technical indicators (RSI, MACD, ADX, Bollinger, MAs)
- 📰 News sentiment (via NLP / VADER)
- 👥 Analyst consensus trends
- 🤖 Monte Carlo–based AI forecasting
- 💵 Adaptive DCA and paper-trading simulations

It’s built for **education, research, and strategy testing** — *not live brokerage execution.*

---

## 💻 How the Dashboard Works

1️⃣ **Enter a stock symbol** (e.g. `MSFT`, `NVDA`, `META`)  
2️⃣ **Choose mode** — short-term swing or long-term investor  
3️⃣ The system instantly:
   - Pulls real market and news data
   - Computes 9+ technical indicators
   - Runs sentiment + forecast models  
4️⃣ Displays your **AI Signal (Buy/Hold/Sell)** with confidence  
5️⃣ Interactive chart shows **Buy Zone**, **Target**, and **Stop** levels  
6️⃣ **"Why this signal"** section explains every factor  
7️⃣ **Forecast AI (5d)** shows the probability-based move range  
8️⃣ **Simulators** help test strategies (DCA, Monte Carlo)
9️⃣ **Analyst Pulse + Macro Dashboard** summarize the broader market view

---

## 📈 Signal Interpretation

| Signal | Meaning | Educational Insight |
|---------|----------|---------------------|
| 🟢 **BUY** | Technical + sentiment trend aligned bullish | Potential short-term or accumulation zone |
| 🟠 **HOLD** | Mixed indicators or neutral momentum | Wait for confirmation |
| 🔴 **SELL** | Overbought or weakening trend | Consider trimming or avoiding new positions |

**Targets & Zones**
- **Buy Zone** — algorithmic dip area (ATR-based)
- **Target** — short-term upside range
- **Stop** — volatility-adjusted downside guardrail

---

## 💰 Paper Trading Strategy

- Track simulated “buys” in your paper-trade log or Excel
- Monitor if price hits the Target or Stop
- Assess signal success over time
- Adjust DCA frequency, position sizing, or stop width

---

## ⚙️ Indicators Used

| Indicator | Purpose |
|------------|----------|
| MA50 / MA200 | Trend direction |
| MACD | Momentum crossover |
| RSI | Overbought / Oversold levels |
| Bollinger Bands | Volatility squeeze or break |
| ADX | Trend strength |
| ATR | Stop/target calibration |
| Sentiment | News tone via AI |
| Analyst Pulse | Consensus from experts |

---

## 🤖 AI Forecast Engine
Runs 1,000 Monte Carlo simulations of recent 120-day returns to estimate:
- **Expected Move (μ)**  
- **Confidence (σ)**  
- **Range (±95%)**

🧮 *Example:*  
> Predicted move: +3.2% in 5d  
> Range: –4.5% → +7.8%  
> Confidence: 68%

---

## 💵 DCA & Monte Carlo Simulators

- **Adaptive DCA Simulator** tests buying on dips + partial selling in rallies  
- **Monte Carlo Future DCA** predicts potential portfolio outcomes  
- Metrics: ROI %, Max Drawdown, Final Value

---

## 🌎 Macro & Analyst Insight

| Section | Description |
|----------|--------------|
| **Macro Dashboard** | VIX, CPI, Unemployment, S&P trend |
| **Analyst Pulse** | Aggregated expert buy/hold/sell sentiment |
| **Market Bias** | Visual summary (Bullish / Bearish / Neutral) |
| **Top Tech Scans** | Daily AI-screened BUY signals from major tech stocks |




---
## ⚖️ Legal Disclaimer
> AISigmaX is for **educational and informational purposes only**.  
> It does **not** constitute financial advice or trading recommendations.  
> All simulations are hypothetical. Investing carries risk — always research independently.
---

### 💬 Support
**AISigmaX** — Powered by MarketSignal LLC  
🌐[www.aisigmax.com](https://www.aisigmax.com)   
📧support@aisigmax.com  
© 2025 MarketMinds LLC. All rights reserved.
""")


# mark tutorial as shown for current session
st.session_state.tutorial_shown = True

# ============================================================
# ✅ Stripe Checkout Return Handler (Success / Cancel)
# ============================================================

# Read URL query parameters after redirect from Stripe
query_params = st.experimental_get_query_params()

# Success page
if "success" in query_params:
    st.markdown("""
    <div style='padding:20px;border-radius:10px;background:#E6FFED;border:2px solid #00B871;margin-bottom:15px;'>
        <h3 style='color:#00B871;'>✅ Payment Successful!</h3>
        <p style='font-size:16px;'>Thank you for subscribing to <strong>AI Stock Signals</strong>.<br>
        You now have full access to <b>AI-based signals, simulators, and forecasts</b>.</p>
        <p style='font-size:15px;'>You can close this message and start exploring your dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

# Cancelled payment page
elif "cancelled" in query_params:
    st.markdown("""
    <div style='padding:20px;border-radius:10px;background:#FFF4E6;border:2px solid #FF9900;margin-bottom:15px;'>
        <h3 style='color:#FF9900;'>⚠️ Payment Cancelled</h3>
        <p style='font-size:16px;'>Your payment was not completed.<br>
        You can upgrade anytime by selecting <b>PRO</b> or <b>ELITE</b> below.</p>
    </div>
    """, unsafe_allow_html=True)





st.markdown("## 💳 Upgrade Your Plan — Unlock Full AI Access")

pricing_html = """
<style>
.pricing-flex {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 18px;
  margin: 30px 0;
}
.pricing-card {
  padding: 22px 18px 18px 18px;
  border-radius: 12px;
  max-width: 320px;
  width: 100%;
  min-width: 220px;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  margin-bottom: 0;
  flex: 1 1 260px;
}
.pricing-card h3 { margin-top: 0; font-size: 1.3em; }
.pricing-card h2 { margin: 10px 0 18px 0; font-size: 2em; }
.pricing-card button, .pricing-card a button {
  width: 100%;
  padding: 12px 0;
  font-size: 1.1em;
  border-radius: 8px;
  border: none;
  font-weight: 600;
  margin-top: 8px;
  cursor: pointer;
}
.pricing-card.free { background: #43e97b; color: #fff; }
.pricing-card.free .current-btn { background: #fff; color: #43e97b; }
.pricing-card.pro { background: #1976d2; color: #fff; }
.pricing-card.pro button { background: #fff; color: #1976d2; }
.pricing-card.elite { background: #8e24aa; color: #fff; }
.pricing-card.elite button { background: #fff; color: #8e24aa; }
@media (max-width: 900px) {
  .pricing-flex { flex-direction: column; align-items: center; }
  .pricing-card { max-width: 98vw; }
}
</style>
<div class="pricing-flex">
  <div class="pricing-card free">
    <h3>Free Tier</h3>
    <p>Access up to 3 tickers/day<br>Basic indicators only</p>
    <h2>$0</h2>
    <button class="current-btn" disabled>Current</button>
  </div>
  <div class="pricing-card pro">
    <h3>Pro</h3>
    <p>Full signals + simulators<br>AI-based Buy/Sell + RSI + DCA</p>
    <h2>$9.99/mo</h2>
    <a href="https://buy.stripe.com/test_14AfZ9akR7dTexO4x5g3602" target="_blank">
      <button>Upgrade to PRO</button>
    </a>
  </div>
  <div class="pricing-card elite">
    <h3>Elite</h3>
    <p>Everything in PRO + Forecast AI<br>Macro Dashboard + Screener Access</p>
    <h2>$29.99/mo</h2>
    <a href="https://buy.stripe.com/test_6oU28j64B1Tz1L23t1g3603" target="_blank">
      <button>Upgrade to ELITE</button>
    </a>
  </div>
</div>
"""
st.markdown(pricing_html, unsafe_allow_html=True)


# ============================================================
# 🔐 Require login BEFORE user can use Ticker / signals
# (But AFTER banner, pricing, etc.)
# ============================================================
#if "user" not in st.session_state:
if st.session_state.user is None:
   
    st.markdown("### 🔐 Login or Create an Account to Use AISigmaX")

    tab_login, tab_signup = st.tabs(["Login", "Signup"])

    # ---------- LOGIN ----------
    with tab_login:
        login_email = st.text_input("Email", key="login_email")
        login_pass = st.text_input("Password", key="login_pass", type="password")


        if st.button("Login"):
            try:
                # fetch user by email
                res = supabase.table("users") \
                    .select("*") \
                    .eq("email", login_email.lower()) \
                    .execute()

                if res.data:
                    user_row = res.data[0]
                    stored_hash = user_row["password_hash"]

                    if bcrypt.checkpw(login_pass.encode(), stored_hash.encode()):
                        st.session_state.user = user_row
                        st.success("✅ Logged in!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password.")
                else:
                    st.error("❌ No account found for that email.")
            except Exception as e:
                st.error(f"Login failed: {e}")

    # ---------- SIGNUP ----------
    with tab_signup:
        signup_email = st.text_input("Signup Email", key="signup_email")
        signup_pass = st.text_input("New Password", key="signup_pass", type="password")


        if st.button("Create Account"):
            try:
                pw_hash = bcrypt.hashpw(signup_pass.encode(), bcrypt.gensalt()).decode()

                res = supabase.table("users").insert({
                    "email": signup_email.lower(),
                    "password_hash": pw_hash,
                    "plan": "free"
                }).execute()

                st.success("🎉 Account created! Please go to Login tab and sign in.")
            except Exception as e:
                st.error(f"Signup failed: {e}")

    # ⛔ stop the rest of the app (ticker, signals, etc.)
    st.stop()

# ============================================================
# 👤 Logged-in user + plan flags
# ============================================================
user = st.session_state.user
current_plan = (user.get("plan") or "free").lower()

is_free  = (current_plan == "free")
is_pro   = (current_plan == "pro")
is_elite = (current_plan == "elite")

st.caption(f"👤 Logged in as **{user['email']}**  •  Plan: **{current_plan.upper()}**")

# Optional logout button (shows once logged in)
if st.button("Logout"):
    st.session_state.user = None
    st.rerun()





c1, c2, c3 = st.columns([2,2,3])
with c1:
    ticker = st.text_input("Ticker", "", placeholder="Enter a stock symbol (e.g., MSFT)").upper().strip()
    
with c2:
    horizon = st.radio("Mode", ["Short-term (Swing)", "Long-term (Investor)"], index=1, horizontal=True)
with c3:
    invest_amount = st.slider("Allocation for simulations ($)", min_value=500, max_value=50_000, step=500, value=10_000)

if not ticker:
    st.markdown("""
        ### 👋 Welcome to AI Stock Signals
        - Enter a symbol above to generate **signals, targets & forecasts**.
        - Example tickers: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `META`, etc.
    """)
    st.stop()

@st.cache_data(ttl=3600)
def get_prev_close(ticker):
    try:
        t = yf.Ticker(ticker)
        return float(t.info.get("previousClose", np.nan))
    except Exception:
        return np.nan

prev_close = get_prev_close(ticker)

# Fallback to daily data if needed
if np.isnan(prev_close):
    df_daily = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
    if not df_daily.empty and len(df_daily) > 1:
        prev_close = float(df_daily["Close"].iloc[-2])

# Get latest price from fast_info or intraday
try:
    t = yf.Ticker(ticker)
    live_price = t.fast_info.get("last_price", np.nan)
except Exception:
    live_price = np.nan

if np.isnan(live_price):
    df_intraday = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
    if not df_intraday.empty:
        live_price = float(df_intraday["Close"].iloc[-1])

if not np.isnan(live_price) and not np.isnan(prev_close):
    current_price = live_price
    current_change = live_price - prev_close
    current_change_pct = (current_change / prev_close) * 100 if prev_close != 0 else 0
else:
    current_price = np.nan
    current_change = 0
    current_change_pct = 0


# ------------------------------ Timeframe ------------------------------
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

# ------------------------------ Data Fetchers ------------------------------
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

@st.cache_data(ttl=3600)
def fetch_major_indices():
    indices = {"Dow Jones": "^DJI", "Nasdaq": "^IXIC", "S&P 500": "^GSPC", "VXN (Nasdaq Volatility)": "^VXN"}
    data = {}
    for name, symbol in indices.items():
        try:
            df = yf.download(symbol, period="1mo", interval="1d", progress=False)
            if df.empty:
                data[name] = None
                continue
            last_row = df.iloc[-1]
            data[name] = {k: float(last_row[k]) for k in ["Open","High","Low","Close","Volume"]}
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
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.rename(columns={series_id: "value"}).dropna()
            df = df[df["value"] != "."]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna()
        except Exception:
            return None

    try:
        cpi_df = fred_csv_last("CPIAUCSL")
        if cpi_df is not None:
            cpi_df = cpi_df.sort_values("DATE")
            last = cpi_df.iloc[-1]["value"]
            prev12 = cpi_df.iloc[-13]["value"] if len(cpi_df) > 13 else np.nan
            macro["cpi_yoy"] = round(((last/prev12) - 1) * 100, 2) if prev12 == prev12 else None
            macro["cpi_date"] = cpi_df.iloc[-1]["DATE"].date().isoformat()
    except Exception:
        macro["cpi_yoy"], macro["cpi_date"] = None, None
    try:
        un_df = fred_csv_last("UNRATE")
        if un_df is not None:
            un_df = un_df.sort_values("DATE")
            macro["unemp_rate"] = round(float(un_df.iloc[-1]["value"]), 2)
            macro["unemp_date"] = un_df.iloc[-1]["DATE"].date().isoformat()
    except Exception:
        macro["unemp_rate"], macro["unemp_date"] = None, None

    if macro.get("cpi_yoy") is None: macro["cpi_yoy"] = 3.2
    if macro.get("unemp_rate") is None: macro["unemp_rate"] = 3.8
    return macro

# ------------------------------ Indicators / Signals ------------------------------
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
    # DI/ADX (approx)
    plus_di  = 100 * (plus_dm.rolling(14, min_periods=1).sum()  / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["ADX"] = dx.rolling(14, min_periods=1).mean()
    # Band width
    try:
        width = (out["BB_Up"] - out["BB_Low"]) / c.replace(0, np.nan)
        if isinstance(width, pd.DataFrame):
            width = width.iloc[:, 0]
        out["BB_Width"] = pd.Series(width, index=df.index).astype(float).fillna(0)
    except Exception:
        out["BB_Width"] = pd.Series(0.0, index=df.index)
    out["Close"] = c
    return out.bfill().ffill()

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

@st.cache_data(ttl=86400)
def analyst_pulse(ticker: str):
    """Analyst consensus — rate-limit safe, silent fallback."""
    try:
        time.sleep(random.uniform(0.25, 0.7))
        t = yf.Ticker(ticker)

        # Try historical recommendations
        try:
            rec = getattr(t, "recommendations", None)
            if rec is not None and not rec.empty:
                df = rec.tail(200).copy()
                df.columns = [c.lower() for c in df.columns]
                col = next((c for c in ["to grade", "action"] if c in df.columns), None)
                if col:
                    grades = df[col].astype(str).str.lower()
                    total = len(grades)
                    buy_terms = ["buy", "strong buy", "outperform", "overweight", "add", "accumulate", "top pick"]
                    hold_terms = ["hold", "neutral", "market perform", "equal weight", "sector perform"]
                    sell_terms = ["sell", "underperform", "underweight", "reduce", "negative"]
                    buy = grades.str.contains("|".join(buy_terms)).sum()
                    hold = grades.str_contains("|".join(hold_terms)).sum() if hasattr(grades, "str_contains") else grades.str.contains("|".join(hold_terms)).sum()
                    sell = grades.str.contains("|".join(sell_terms)).sum()
                    return {"buy": buy/total if total else None, "hold": hold/total if total else None,
                            "sell": sell/total if total else None, "neutral": (hold/total if total else None), "samples": total}
        except Exception:
            pass

        # recommendations_summary
        trend = getattr(t, "recommendations_summary", None)
        row = None
        if trend is not None:
            if isinstance(trend, pd.DataFrame):
                row = trend.iloc[0].to_dict() if "strongBuy" in trend.columns else trend.to_dict(orient="records")[0]
            elif isinstance(trend, dict):
                row = trend
        if row:
            total = sum(v for v in row.values() if isinstance(v, (int, float)))
            if total > 0:
                buy = (row.get("buy", 0) + row.get("strongBuy", 0)) / total
                hold = row.get("hold", 0) / total
                sell = (row.get("sell", 0) + row.get("strongSell", 0)) / total
                return {"buy": buy, "hold": hold, "sell": sell, "neutral": hold, "samples": total}

        # Yahoo JSON
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=recommendationTrend"
            r = requests.get(url, timeout=10)
            if r.ok:
                js = r.json()
                trend_list = js.get("quoteSummary", {}).get("result", [{}])[0].get("recommendationTrend", {}).get("trend", [])
                if trend_list:
                    latest = trend_list[-1]
                    total = sum(latest.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
                    if total > 0:
                        buy = (latest.get("buy", 0) + latest.get("strongBuy", 0)) / total
                        hold = latest.get("hold", 0) / total
                        sell = (latest.get("sell", 0) + latest.get("strongSell", 0)) / total
                        return {"buy": buy, "hold": hold, "sell": sell, "neutral": hold, "samples": total}
        except Exception:
            pass

        return {"buy": None, "hold": None, "sell": None, "neutral": None, "samples": 0}
    except Exception:
        return {"buy": None, "hold": None, "sell": None, "neutral": None, "samples": 0}

def market_confidence(sentiment: float, buy_ratio: float | None):
    sent_norm = (sentiment + 1) / 2
    conf = 0.6 * sent_norm + 0.4 * (buy_ratio if buy_ratio is not None else 0.5)
    return max(0, min(1, conf))

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
    if last.get("BB_Width", 0) > 0.12: score -= 0.2
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
    if last["MA50"] > last["MA200"]:
        reasons.append("✅ **Uptrend** — MA50 above MA200 (long-term strength).")
    else:
        reasons.append("⚠️ **Downtrend** — MA50 below MA200 (bearish bias).")
    if last["MACD"] > last["MACD_Signal"]:
        reasons.append("✅ **MACD bullish crossover** — momentum improving.")
    else:
        reasons.append("⚠️ **MACD bearish** — momentum fading.")
    if last["RSI"] < 30:
        reasons.append("✅ **RSI oversold** (<30) — potential rebound zone.")
    elif last["RSI"] > 70:
        reasons.append("⚠️ **RSI overbought** (>70) — may need cooldown.")
    elif 45 <= last["RSI"] <= 55:
        reasons.append("💤 **RSI neutral** — sideways momentum.")
    bb_width = last.get("BB_Width", 0)
    if bb_width < 0.05:
        reasons.append("🔹 **Bollinger squeeze** — volatility contraction, breakout possible.")
    elif last["Close"] < last["BB_Low"]:
        reasons.append("✅ **Price below lower band** — mean reversion likely.")
    elif last["Close"] > last["BB_Up"]:
        reasons.append("⚠️ **Price above upper band** — extended move, possible pullback.")
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
    reasons.append("🎯 Strategy tuned for **short-term swing** (3–10d)." if "Short" in horizon
                   else "🏦 Strategy tuned for **long-term accumulation** (>3mo).")
    return "\n".join(reasons)

def daily_action_strategy(price, buy_zone, target_up, stop_loss, signal, invest_amount, shares_held=0, cash=0):
    action = "HOLD"
    shares = 0.0
    msg = ""
    buy_amount = invest_amount * 0.25
    if signal == "BUY" or price < buy_zone:
        action = "BUY"
        shares = buy_amount / price
        msg = f"📈 **Buy Signal:** Buy {shares:.2f} shares at ${price:.2f}"
    elif price >= target_up and shares_held > 0:
        action = "SELL"
        shares = shares_held * 0.5
        msg = f"🏁 **Sell Signal:** Sell {shares:.2f} shares at ${price:.2f} (Target: ${target_up:.2f})"
    elif price <= stop_loss and shares_held > 0:
        action = "STOP"
        shares = shares_held
        msg = f"🛑 **Stop Signal:** Sell all ({shares:.2f}) shares at ${price:.2f} (Stop: ${stop_loss:.2f})"
    else:
        msg = "🤔 **Hold:** No action today. Wait for a new signal or price movement."
    return {"action": action, "price": price, "shares": shares, "msg": msg}

def ai_forecast(df: pd.DataFrame, ind: pd.DataFrame):
    r = df["Close"].pct_change().dropna()
    if isinstance(r, pd.DataFrame): r = r.iloc[:, 0]
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 30:
        return {"pred_move": 0.0, "conf": 0.0, "range": None}
    r_hist = r.tail(120).values.flatten()
    sims = np.random.choice(r_hist, size=(1000, 5), replace=True).sum(axis=1)
    mu = float(np.mean(sims))
    sd = float(np.std(sims))
    low = mu - 1.96 * sd
    high = mu + 1.96 * sd
    conf = float(min(1.0, abs(mu) / (sd + 1e-9)))
    return {"pred_move": mu, "conf": conf, "range": (low, mu, high)}

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
        fig.add_hline(y=buy_zone,  line_color="dodgerblue", line_dash="dash", annotation_text="Buy Zone", row=1, col=1)
        fig.add_hline(y=target,    line_color="seagreen",   line_dash="dash", annotation_text="Target",   row=1, col=1)
        fig.add_hline(y=stop_loss, line_color="crimson",    line_dash="dash", annotation_text="Stop",     row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"],        name="MACD",   line=dict(color="purple")), 2,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", line=dict(color="orange")), 2,1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_Hist"], name="Hist", marker_color="gray", opacity=0.45), 2,1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI"], name="RSI", line=dict(color="teal")), 3,1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.update_layout(height=820, template="plotly_white", title=f"{ticker} — Technical Dashboard")
    return fig

# ------------------------------ Main Flow ------------------------------
df = fetch_prices_tf(ticker, period, interval)
if df is None or df.empty:
    st.error("No data found. Try another symbol.")
    st.stop()

ind = compute_indicators(df)
macro = fetch_macro()
headlines, news_sent = fetch_news_and_sentiment(ticker)
decision, color, score = generate_signal(ind, news_sent, horizon)
pulse = analyst_pulse(ticker)
conf_overall = market_confidence(news_sent, pulse["buy"])
ai = ai_forecast(df, ind)

# ------------------------------ Signal Card (Top) ------------------------------


st.markdown(f"## ✅ Signal: **{decision}**  (Score {score:+.2f}, News {news_sent:+.2f})")
st.progress(conf_overall, text=f"Market Confidence {int(conf_overall*100)}% — sentiment/analyst blend")

# Snapshot metrics
last = ind.iloc[-1]
cA, cB, cC, cD, cE, cF = st.columns(6)
#cA.metric("Price", f"${last['Close']:.2f}")

# Calculate daily change
last = ind.iloc[-1]
prev = ind.iloc[-2] if len(ind) > 1 else last

price = last["Close"]
change = price - prev["Close"]
change_pct = (change / prev["Close"]) * 100 if prev["Close"] != 0 else 0

cA, cB, cC, cD, cE, cF = st.columns(6)
cA.metric("Price", f"${price:.2f}", delta=f"{change:+.2f} ({change_pct:+.2f}%)")

cB.metric("RSI (14)", f"{last['RSI']:.1f}")
cC.metric("MACD", f"{last['MACD']:.2f}")
cD.metric("ADX", f"{last['ADX']:.1f}")
cE.metric("ATR (14)", f"{last['ATR']:.2f}")


st.metric("Signal Strength", f"{int(confidence_from_score(score)*100)}%", delta=f"{score:+.2f}")

target_up = last["Close"] + 2.0*last["ATR"]
buy_zone  = last["Close"] - 1.5*last["ATR"]
stop_loss = last["Close"] - 2.5*last["ATR"]
st.write(f"📈 **Target (≈5d)**: ${target_up:.2f}  🟦 **Buy zone**: ${buy_zone:.2f}  🛑 **Stop**: ${stop_loss:.2f}")

# One-day action suggestion (immediately after target/stop)
shares_held = 0  # or your actual shares held if you track it
cash = invest_amount
today_action = daily_action_strategy(
    price, buy_zone, target_up, stop_loss, decision, invest_amount, shares_held, cash
)
st.markdown(today_action["msg"])

# ------------------------------ Chart ------------------------------
st.plotly_chart(plot_dashboard(ind, ticker, zones=True), use_container_width=True)

# ------------------------------ Why This Signal ------------------------------
st.markdown("### 🧩 Why this signal")
st.markdown(explain_signal_verbose(ind, news_sent, decision, horizon))

# ------------------------------ Forecast AI ------------------------------
#st.markdown("### 🤖 Forecast AI (5-day)")
#st.write(f"Predicted Move (avg): {ai['pred_move']*100:+.2f}%")
#if ai["range"] is not None and not any(np.isnan(ai["range"])):
    #lo, mu, hi = ai["range"]
    #st.write(f"Expected range in 5d: {lo*100:+.2f}% — {mu*100:+.2f}% — {hi*100:+.2f}%")
#else:
    #st.info("Not enough recent data for a reliable range forecast.")
#st.metric("AI Confidence", f"{int(ai['conf']*100)}%")

st.markdown("### 🤖 Forecast AI (5-day)")

if is_free:
    st.info("Forecast AI is available on PRO and ELITE plans. Upgrade above to unlock probability-based 5-day forecasts.")
else:
    st.write(f"Predicted Move (avg): {ai['pred_move']*100:+.2f}%")
    if ai["range"] is not None and not any(np.isnan(ai["range"])):
        lo, mu, hi = ai["range"]
        st.write(f"Expected range in 5d: {lo*100:+.2f}% — {mu*100:+.2f}% — {hi*100:+.2f}%")
    else:
        st.info("Not enough recent data for a reliable range forecast.")
    st.metric("AI Confidence", f"{int(ai['conf']*100)}%")


# ------------------------------ Simulators (Tabs) ------------------------------
st.markdown("### 🧪 Simulation Tools")
tab1, tab2, tab3 = st.tabs(["Daily Action", "Adaptive DCA", "Monte Carlo"])



with tab1:
    st.write("One-day action suggestion (for learning/demo):")
    day_action = daily_action_strategy(price, buy_zone, target_up, stop_loss, decision, invest_amount, shares_held=0, cash=invest_amount)
    st.markdown(day_action["msg"])

with tab2:
    st.write("Adaptive DCA (oversold/momentum-aware with partial take-profit):")
    def adaptive_dca_simulator(df: pd.DataFrame, ind: pd.DataFrame, cash_start: float):
        df, ind = df.align(ind, join="inner", axis=0)
        cash, shares = float(cash_start), 0.0
        trades, equity_curve = [], []
        peak_equity, halt_buys = cash_start, False

        for dt in df.index:
            px = float(df.loc[dt, "Close"])
            rsi, macd, macds = float(ind.loc[dt, "RSI"]), float(ind.loc[dt, "MACD"]), float(ind.loc[dt, "MACD_Signal"])
            ma20, ma50 = float(ind.loc[dt, "MA20"]), float(ind.loc[dt, "MA50"])
            bb_low, atr = float(ind.loc[dt, "BB_Low"]), float(ind.loc[dt, "ATR"])

            if not halt_buys:
                momentum_buy = (macd > macds and ma20 > ma50)
                oversold_buy = (rsi < 45) or (px < bb_low)
                alloc = 0.0
                if momentum_buy or oversold_buy:
                    if rsi < 25: alloc = 0.30
                    elif rsi < 35: alloc = 0.20
                    elif rsi < 45: alloc = 0.10
                invest = cash * alloc
                if invest > 0:
                    buy_shares = invest / px
                    shares += buy_shares
                    cash -= invest
                    trades.append({"date": dt.strftime("%Y-%m-%d"), "side": "BUY", "price": round(px,2),
                                   "invested": round(invest,2), "shares": round(buy_shares,6)})

            target_price = float(ind["Close"].iloc[-1] + 2*ind["ATR"].iloc[-1])
            if shares > 0 and px >= target_price:
                sell_shares = shares * 0.20
                proceeds = sell_shares * px
                shares -= sell_shares
                cash += proceeds
                trades.append({"date": dt.strftime("%Y-%m-%d"), "side": "SELL", "price": round(px,2),
                               "invested": -round(proceeds,2), "shares": -round(sell_shares,6)})

            equity = float(shares * px + cash)
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            dd_pct = (equity - peak_equity) / (peak_equity if peak_equity else 1)
            if dd_pct < -0.30:
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

    sim = adaptive_dca_simulator(df, ind, invest_amount)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Portfolio Value", f"${safe_float(sim.get('final_value')):,.2f}")
    c2.metric("Total Invested", f"${safe_float(sim.get('total_invested')):,.2f}")
    c3.metric("ROI", f"{safe_float(sim.get('roi_pct')):.1f}%")
    c4.metric("Max Drawdown", f"{safe_float(sim.get('max_drawdown_pct')):.1f}%")
    if not sim["trades"].empty:
        st.dataframe(sim["trades"], use_container_width=True)
    else:
        st.info("No trades executed in this period by adaptive rules.")

with tab3:
    st.write("Monte Carlo forward outcomes (bootstrap past returns):")
    periods_map = {"1D":1,"1W":5,"1M":21,"3M":63,"6M":126,"YTD":180,"1Y":252,"2Y":504,"5Y":1260,"10Y":2520,"ALL":252}
    days = periods_map.get(tf, 21)

    def simulate_future_prices(df, days=10, n_sims=1000):
        returns_series = df["Close"].pct_change().dropna()
        if isinstance(returns_series, pd.DataFrame): returns_series = returns_series.iloc[:, 0]
        elif isinstance(returns_series, np.ndarray) and returns_series.ndim > 1:
            returns_series = pd.Series(returns_series.flatten())
        elif not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)
        returns = pd.to_numeric(returns_series, errors="coerce").values
        returns = returns[~np.isnan(returns)]
        if len(returns) < max(10, days // 3):
            return None
        last_price = float(df["Close"].iloc[-1])
        sims = []
        for _ in range(n_sims):
            sampled_returns = np.random.choice(returns, size=days, replace=True)
            price_path = [last_price]
            for r in sampled_returns:
                price_path.append(price_path[-1] * (1 + r))
            sims.append(price_path[1:])
        return np.array(sims)

    sim_prices = simulate_future_prices(df, days=days, n_sims=1000)
    if sim_prices is None:
        st.info("Not enough data for reliable simulation at this timeframe.")
    else:
        predicted_prices = sim_prices[:, -1]
        mean_price = np.mean(predicted_prices); median_price = np.median(predicted_prices)
        low_price  = np.percentile(predicted_prices, 2.5); high_price = np.percentile(predicted_prices, 97.5)
        buy_price = float(df["Close"].iloc[-1])
        expected_gain = mean_price - buy_price
        expected_gain_pct = (expected_gain / buy_price) * 100 if buy_price != 0 else 0
        st.markdown(f"**Current price:** ${buy_price:.2f}  |  **Mean:** ${mean_price:.2f}  |  **Median:** ${median_price:.2f}  |  **95% range:** ${low_price:.2f} — ${high_price:.2f}")
        st.write(f"**Expected gain/loss per share:** ${expected_gain:+.2f} ({expected_gain_pct:+.2f}%)")

        # Small histogram
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.hist(predicted_prices, bins=30, alpha=0.8)
        ax.set_title(f"Distribution of Simulated {days}-Day Prices")
        ax.set_xlabel("Price")
        ax.set_ylabel("Sims")
        plt.tight_layout()
        st.pyplot(fig)

# ------------------------------ Macro & Fundamentals ------------------------------
st.markdown("### 🌎 Market & Fundamentals")

m1, m2, m3, m4 = st.columns(4)
macro_vals = fetch_macro()
m1.metric("VIX (volatility)", f"{macro_vals['vix_last']:.2f}" if macro_vals["vix_last"] is not None else "—")
m2.metric("S&P 5d vs 20d", f"{macro_vals['spx_5d_vs_20d']:+.2f}%" if macro_vals["spx_5d_vs_20d"] is not None else "—",
          macro_vals["spx_trend"] or "")
m3.metric("CPI YoY", f"{macro_vals.get('cpi_yoy', 0):.2f}%")
m4.metric("Unemployment", f"{macro_vals.get('unemp_rate', 0):.2f}%")

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
            idx_cols[i].metric(f"{name} Close", "—")
            idx_cols[i].metric(f"{name} High", "—")
            idx_cols[i].metric(f"{name} Low", "—")
            idx_cols[i].metric(f"{name} Volume", "—")

with st.expander("📊 Stock Fundamentals", expanded=False):
    fcols = st.columns(12)
    labels = ["Open","High","Low","Volume","P/E","Market Cap","52w High","52w Low","Avg Vol","Yield","Beta","EPS"]
    kinds  = [None,None,None,"vol",None,"cap",None,None,"vol","pct",None,None]
    for i,(lab,kind) in enumerate(zip(labels, kinds)):
        val = fund.get(lab)
        fcols[i].metric(lab, human_fmt(val, kind=kind))

# ------------------------------ Analyst Pulse ------------------------------
def render_analyst_pulse(pulse: dict):
    if not pulse or pulse.get("samples", 0) <= 0:
        st.info("No analyst data available.")
        return
    buy = pulse.get("buy") or 0; hold = pulse.get("hold") or 0; sell = pulse.get("sell") or 0
    total = max(buy + hold + sell, 1e-9)
    buy_pct, hold_pct, sell_pct = [round(x / total * 100, 1) for x in (buy, hold, sell)]
    accent = "#28a745" if (buy>hold and buy>sell) else ("#dc3545" if (sell>buy and sell>hold) else "#f0ad4e")
    mood   = "Bullish" if accent=="#28a745" else ("Bearish" if accent=="#dc3545" else "Neutral")
    st.markdown(f"### 🧭 <span style='color:{accent};'>Analyst Pulse — {mood}</span>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color:#fff;border:1.5px solid {accent};border-radius:8px;padding:6px 10px;margin-top:4px;box-shadow:0 0 6px 0 {accent}22;'>
        <div style='display:flex;align-items:center;gap:10px;'>
            <div style='font-size:13px;color:#555;white-space:nowrap;'><b>Analyst Pulse</b> • {pulse['samples']} ratings</div>
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

render_analyst_pulse(pulse)

# ------------------------------ Tech Screener ------------------------------
st.markdown("## 📋 Today's Top Tech Stock Buy Candidates")
tech_watchlist = [
    "AAPL","MSFT","GOOGL","GOOG","META","AMZN","NVDA","TSLA","CRM",
    "ADBE","ORCL","INTC","AMD","CSCO","NFLX","AVGO","QCOM","IBM","SHOP","UBER"
]
buy_candidates = []
for scan_t in tech_watchlist:
    try:
        scan_df = fetch_prices_tf(scan_t, period, interval)
        if scan_df is None or len(scan_df) < 30: continue
        scan_ind = compute_indicators(scan_df)
        _, scan_sent = fetch_news_and_sentiment(scan_t)
        sig, _, sc = generate_signal(scan_ind, scan_sent, horizon)
        last_s = scan_ind.iloc[-1]; price_s = last_s["Close"]
        buy_z   = price_s - 1.5 * last_s["ATR"]
        target_ = price_s + 2.0 * last_s["ATR"]
        stop_   = price_s - 2.5 * last_s["ATR"]
        earn_dt = fetch_earnings_date(scan_t)
        if sig == "BUY" and price_s <= buy_z * 1.05:
            buy_candidates.append({
                "Ticker": scan_t,
                "Price": f"${price_s:.2f}",
                "Score": sc,
                "Buy Zone": f"${buy_z:.2f}",
                "Target": f"${target_:.2f}",
                "Stop": f"${stop_:.2f}",
                "Earnings": earn_dt,
                "News Sentiment": f"{scan_sent:+.2f}"
            })
    except Exception:
        continue
buy_candidates = sorted(buy_candidates, key=lambda x: x["Score"], reverse=True)
if buy_candidates:
    st.dataframe(pd.DataFrame(buy_candidates), use_container_width=True)
else:
    st.info("No strong tech stock buy candidates found today based on your criteria.")

# ------------------------------ News ------------------------------
with st.expander("🗞️ Latest Headlines", expanded=False):
    if not headlines:
        st.write("No headlines available.")
    else:
        for h in headlines:
            title = h["title"]; url = h["url"]; src = h.get("source",""); pub = h.get("published","")
            nice = pub[:10] if pub else ""
            st.markdown(f"- [{title}]({url}) — *{src}* {('• '+nice) if nice else ''}")

# ------------------------------ Learn & Disclaimer ------------------------------
with st.expander("📘 Learn: Indicators, Patterns & AI Logic", expanded=False):
    st.markdown("""
**What you’re seeing**
- **Signal** blends trend (MA, ADX), momentum (RSI, MACD), extremes (Bollinger), and news sentiment.  
- **Forecast AI** uses bootstrap/Monte Carlo of historical returns to estimate a 5-day range.  
- **Simulators** model daily action and adaptive DCA with partial take-profit.

**Indicator notes**  
- RSI: <30 oversold, >70 overbought.  
- MACD: momentum/trend crossovers.  
- Bollinger: ±2σ around 20D mean; squeeze can precede breakouts.  
- ADX: trend strength (>25 strong).  
- ATR: volatility; used for target/stop bands.  
""")

# ... your dashboard code ...
# ============================================================
#  CHAT MODULE – AISigmaX Assistant (Google + Technicals)
# ============================================================

import re
import requests
import streamlit as st


# ------------------------------------------------------------
# GOOGLE SEARCH API
# ------------------------------------------------------------
def google_search_response(q: str) -> str:
    api_key = st.secrets.get("GOOGLE_API_KEY")
    cx = st.secrets.get("GOOGLE_CX")

    if not api_key or not cx:
        return "⚠️ Google Search is not configured. Add GOOGLE_API_KEY + GOOGLE_CX to Secrets."

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": q}

    try:
        response = requests.get(url, params=params, timeout=10).json()
    except Exception as e:
        return f"⚠️ Google Search failed:\n{e}"

    if "items" not in response:
        return "⚠️ No Google results found."

    msg = ""
    for item in response["items"][:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        msg += f"🔎 **{title}**\n{snippet}\n👉 <{link}>\n\n"

    return msg.strip()



# ------------------------------------------------------------
# TICKER EXTRACTION
# ------------------------------------------------------------
def extract_ticker(text: str):
    text_up = text.upper()
    tokens = re.findall(r"\b[A-Z]{2,5}\b", text_up)

    # Prefer current dashboard ticker
    if ticker and ticker.upper() in tokens:
        return ticker.upper()

    # Common US tickers
    known = {
        "AAPL","TSLA","AMZN","MSFT","META",
        "GOOG","GOOGL","NVDA","NFLX","IBM",
        "AMD","INTC","ORCL","CRM","SHOP","UBER"
    }
    for tkn in tokens:
        if tkn in known:
            return tkn

    # Names → tickers
    names = {
        "TESLA": "TSLA",
        "APPLE": "AAPL",
        "AMAZON": "AMZN",
        "MICROSOFT": "MSFT",
        "GOOGLE": "GOOG",
        "ALPHABET": "GOOG",
        "META": "META",
        "NETFLIX": "NFLX",
        "NVIDIA": "NVDA",
    }
    for name, sym in names.items():
        if name in text_up:
            return sym

    return None



# ------------------------------------------------------------
# TECHNICAL SUMMARY USING DASHBOARD DATA
# ------------------------------------------------------------
def analyze_ticker(requested_ticker: str) -> str:
    global_ticker = globals().get("ticker")
    ind = globals().get("ind")
    decision = globals().get("decision", "HOLD")

    # Only allow technical indicators for dashboard ticker
    if requested_ticker.upper() != global_ticker.upper():
        return (
            f"📊 I can only show technical indicators for the dashboard symbol **{global_ticker.upper()}**.\n\n"
            f"To analyze indicators for **{requested_ticker}**, change the ticker at the top."
        )

    if ind is None or ind.empty:
        return "⚠️ Technical data not ready yet."

    last = ind.iloc[-1]

    rsi_val = float(last["RSI"])
    macd_val = float(last["MACD"])
    adx_val = float(last["ADX"])
    ma50 = float(last["MA50"])
    ma200 = float(last["MA200"])

    trend = "Uptrend (MA50 > MA200)" if ma50 > ma200 else "Downtrend (MA50 < MA200)"
    momentum = "Strong" if adx_val > 25 else "Weak / Range-bound"

    return f"""
📊 **AISigmaX Technical Summary for {requested_ticker.upper()}**

• RSI: **{rsi_val:.1f}**  
• MACD: **{macd_val:.2f}**  
• Trend: **{trend}**  
• Momentum (ADX): **{momentum} – ADX {adx_val:.1f}**  
• AI Signal: **{decision.upper()}**
""".strip()



# ------------------------------------------------------------
# ROUTER — DECIDE RESPONSE TYPE
# ------------------------------------------------------------
def aisigmax_reply(user_msg: str) -> str:
    text = user_msg.lower()

    # 1) Company / Fundamentals / Overview
    company_keywords = [
        "analyse", "analyze", "analysis",
        "company", "overview", "summary",
        "about", "profile", "business"
    ]
    if any(k in text for k in company_keywords):
        t = extract_ticker(user_msg)
        if t:
            return google_search_response(f"{t} company overview financials news")
        return google_search_response(user_msg)

    # 2) Educational questions
    if any(k in text for k in ["what is", "explain", "meaning", "how does"]):
        return google_search_response(user_msg)

    # 3) Earnings / news
    if any(k in text for k in ["financial", "earnings", "news", "revenue"]):
        t = extract_ticker(user_msg)
        if t:
            return google_search_response(f"{t} stock financials earnings news")
        return google_search_response(user_msg)

    # 4) Technical analysis
    t = extract_ticker(user_msg)
    if t:
        return analyze_ticker(t)

    # 5) Default → Google
    return google_search_response(user_msg)



# ============================================================
#  UI — CHATBOX DISPLAY WITH AUTOMATIC SCROLL
# ============================================================

st.markdown("### 💬 Chat with AISigmaX Assistant")

# CSS
st.markdown(
    """
<style>
.chat-box {
    max-height: 440px;
    overflow-y: auto;
    padding: 12px;
    border-radius: 12px;
    background: #f7f9fc;
    border: 1px solid #d0d7e3;
    margin-bottom: 12px;
}
.msg-user {
    background: #dbeafe;
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #3b82f6;
}
.msg-ai {
    background: #f1f5f9;
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #9ca3af;
}
</style>
""",
    unsafe_allow_html=True,
)

# State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat window
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"<div class='msg-user'>🧑 You:<br>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-ai'>🤖 AISigmaX:<br>{msg}</div>", unsafe_allow_html=True)

# Anchor for scrolling
st.markdown("<div id='end-of-chat'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll JS
st.markdown(
    """
<script>
var elem = window.parent.document.getElementById('end-of-chat');
if (elem) { elem.scrollIntoView({behavior: 'smooth'}); }
</script>
""",
    unsafe_allow_html=True,
)

# Input
user_input = st.chat_input("Ask anything about stocks, indicators, technicals, or finance…")

if user_input:
    reply = aisigmax_reply(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("ai", reply))
    st.rerun()

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()



# Now put your disclaimer after the chat
st.markdown("""
---


<div style='text-align:left; color:gray; font-size:14px; line-height:1.5; margin-top:14px;'>
<b>Disclaimer:</b><br>
AISigmaX is a product of <b>MarketSignal LLC</b>.<br>
This dashboard is for <b>educational and informational purposes only</b> and does not constitute financial advice.<br>
Markets carry risk; always do your own research or consult a licensed financial advisor before investing.<br><br>
&copy; 2025 <b>MarketSignal LLC</b> &mdash; <i>AISigmaX</i>
</div>
    """,
    unsafe_allow_html=True,
)









