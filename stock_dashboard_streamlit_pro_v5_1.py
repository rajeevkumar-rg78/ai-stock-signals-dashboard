# stock_dashboard_streamlit_pro_v5_1.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests, feedparser

st.set_page_config(page_title="AI Stock Signals Dashboard v5.1", layout="wide")
st.title("📊 AI Stock Signals Dashboard — PRO v5.1")
st.caption("Integrating technicals, fundamentals, analyst opinions & sentiment (5-year backtest).")

# ------------------------------------------------------------
# User input
# ------------------------------------------------------------
ticker = st.text_input("Enter stock ticker (e.g. AAPL, NVDA, TSLA):", "AAPL").upper()
mode = st.radio("Trading Style", ["Short-Term", "Long-Term"], index=1, horizontal=True)

# ------------------------------------------------------------
# Data fetch
# ------------------------------------------------------------
@st.cache_data(ttl=7200)
def fetch_prices(ticker, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty: return None
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index = df.index.tz_localize(None)
    return df

# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------
def compute_indicators(df):
    out = pd.DataFrame(index=df.index)
    c,h,l,v = df["Close"], df["High"], df["Low"], df["Volume"]
    out["MA20"]=c.rolling(20).mean(); out["MA50"]=c.rolling(50).mean(); out["MA200"]=c.rolling(200).mean()
    delta=c.diff(); up=delta.clip(lower=0); down=-delta.clip(upper=0)
    avg_gain=up.ewm(alpha=1/14,min_periods=14).mean(); avg_loss=down.ewm(alpha=1/14,min_periods=14).mean()
    rs=avg_gain/(avg_loss+1e-9); out["RSI"]=100-(100/(1+rs))
    ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean()
    out["MACD"]=ema12-ema26; out["MACD_Signal"]=out["MACD"].ewm(span=9).mean()
    out["MACD_Hist"]=out["MACD"]-out["MACD_Signal"]
    bb_mid=c.rolling(20).mean(); bb_std=c.rolling(20).std()
    out["BB_Up"]=bb_mid+2*bb_std; out["BB_Low"]=bb_mid-2*bb_std
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    out["ATR"]=tr.rolling(14).mean()
    
    # --- ADX (trend strength, safe flatten fix) ---
    up_move = h.diff()
    down_move = -l.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # 🔒 Force flatten and cast to float64 for Streamlit Cloud
    plus_dm = pd.Series(np.ravel(plus_dm).astype(float), index=df.index)
    minus_dm = pd.Series(np.ravel(minus_dm).astype(float), index=df.index)
    
    atr_smooth = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / (atr_smooth + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["ADX"] = dx.rolling(14, min_periods=1).mean()

    
    atr_smooth=tr.rolling(14).mean()
    plus_di=100*(plus_dm.rolling(14).sum()/(atr_smooth+1e-9))
    minus_di=100*(minus_dm.rolling(14).sum()/(atr_smooth+1e-9))
    dx=(abs(plus_di-minus_di)/(plus_di+minus_di+1e-9))*100
    out["ADX"]=dx.rolling(14).mean()
    vol_ma=v.rolling(20).mean(); out["Vol_Spike"]=(v>2*vol_ma).astype(int)
    out["Close"]=c; return out.bfill().ffill()

# ------------------------------------------------------------
# Fundamentals
# ------------------------------------------------------------
@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    t=yf.Ticker(ticker)
    info=t.fast_info; stats=t.info
    return dict(
        fwdPE=stats.get("forwardPE"), trailPE=stats.get("trailingPE"),
        mcap=stats.get("marketCap"), sector=stats.get("sector"),
        grossM=stats.get("grossMargins"), revG=stats.get("revenueGrowth"),
        rec=stats.get("recommendationMean")
    )

# ------------------------------------------------------------
# News & sentiment
# ------------------------------------------------------------
def get_sentiment(ticker):
    feeds=[f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
           f"https://www.marketwatch.com/rss/headlines.asp?symbol={ticker}"]
    analyzer=SentimentIntensityAnalyzer(); scores=[]
    for url in feeds:
        d=feedparser.parse(url)
        for e in d.entries[:10]:
            text=f"{e.title} {e.get('summary','')}"
            scores.append(analyzer.polarity_scores(text)["compound"])
    return float(np.mean(scores)) if scores else 0.0

# ------------------------------------------------------------
# Signal logic
# ------------------------------------------------------------
def generate_signal(ind,fund,news_s,mode):
    last=ind.iloc[-1]; score=0
    if last["MA20"]>last["MA50"]:score+=1
    if last["MA50"]>last["MA200"]:score+=1
    if last["ADX"]>25:score+=1
    if last["RSI"]<30:score+=1
    elif last["RSI"]>70:score-=1
    if last["MACD"]>last["MACD_Signal"]:score+=1
    else:score-=1
    if last["Close"]<last["BB_Low"]:score+=1
    elif last["Close"]>last["BB_Up"]:score-=1
    if last["Vol_Spike"]:score+=0.5
    if fund.get("fwdPE") and fund["fwdPE"]<25:score+=1
    if fund.get("rec") and fund["rec"]<2.5:score+=1
    score+=news_s
    if mode=="Short-Term":th_buy,th_sell=3.0,-2.0
    else:th_buy,th_sell=4.0,-2.5
    if score>=th_buy:return "BUY","green",score
    elif score<=th_sell:return "SELL","red",score
    else:return "HOLD","orange",score

# ------------------------------------------------------------
# Backtest preview
# ------------------------------------------------------------
# ------------------------------------------------------------
# Backtest preview (safe 1D fix)
# ------------------------------------------------------------
def backtest(df, ind):
    sig = (ind["RSI"] < 35).astype(int) - (ind["RSI"] > 65).astype(int)
    nxt = df["Close"].shift(-5)
    ret = (nxt - df["Close"]) / df["Close"]

    # Ensure both are 1D arrays to prevent alignment error
    sig = np.ravel(sig.values)
    ret = np.ravel(ret.values)

    mask = sig != 0
    if mask.sum() < 10:
        return 0.0

    acc = np.mean(np.sign(ret[mask]) == sig[mask])
    return round(acc * 100, 1)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_all(ind,ticker):
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.5,0.25,0.25],
                      subplot_titles=("Price/MAs/Bollinger","MACD","RSI"))
    fig.add_trace(go.Scatter(x=ind.index,y=ind["Close"],name="Close",line=dict(color="blue")),1,1)
    fig.add_trace(go.Scatter(x=ind.index,y=ind["MA50"],name="MA50",line=dict(color="orange")),1,1)
    fig.add_trace(go.Scatter(x=ind.index,y=ind["MA200"],name="MA200",line=dict(color="green")),1,1)
    fig.add_trace(go.Scatter(x=ind.index,y=ind["BB_Up"],name="BB Upper",line=dict(color="gray",dash="dot")),1,1)
    fig.add_trace(go.Scatter(x=ind.index,y=ind["BB_Low"],name="BB Lower",line=dict(color="gray",dash="dot")),1,1)
    fig.add_trace(go.Scatter(x=ind.index,y=ind["MACD"],name="MACD",line=dict(color="purple")),2,1)
    fig.add_trace(go.Scatter(x=ind.index,y=ind["MACD_Signal"],name="Signal",line=dict(color="orange")),2,1)
    fig.add_trace(go.Bar(x=ind.index,y=ind["MACD_Hist"],name="Hist",marker_color="gray",opacity=0.5),2,1)
    fig.add_trace(go.Scatter(x=ind.index,y=ind["RSI"],name="RSI",line=dict(color="teal")),3,1)
    fig.add_hline(y=70,line_dash="dot",line_color="red",row=3,col=1)
    fig.add_hline(y=30,line_dash="dot",line_color="green",row=3,col=1)
    fig.update_layout(height=800,title=f"{ticker} — Technical Dashboard (5y)",template="plotly_white")
    return fig

# ------------------------------------------------------------
# Explanation
# ------------------------------------------------------------
def explain(ind,fund,news,signal):
    last=ind.iloc[-1]; rs=[]
    rs.append("MA trend up" if last["MA50"]>last["MA200"] else "MA trend down")
    rs.append("RSI low → oversold" if last["RSI"]<35 else ("RSI high → overbought" if last["RSI"]>65 else "RSI neutral"))
    rs.append("MACD bullish" if last["MACD"]>last["MACD_Signal"] else "MACD bearish")
    if fund.get("fwdPE"): rs.append(f"Forward P/E {fund['fwdPE']:.1f}")
    if fund.get("rec"): rs.append(f"Analyst consensus {fund['rec']:.1f}")
    if news>0.1: rs.append("Positive news sentiment")
    elif news<-0.1: rs.append("Negative news sentiment")
    return f"**Why {signal}:** " + ", ".join(rs)

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
if ticker:
    df=fetch_prices(ticker)
    if df is None: st.error("No data found."); st.stop()
    ind=compute_indicators(df)
    fund=get_fundamentals(ticker)
    news=get_sentiment(ticker)
    signal,color,score=generate_signal(ind,fund,news,mode)
    acc=backtest(df,ind)
    last=ind.iloc[-1]

    col1,col2,col3,col4,col5=st.columns(5)
    col1.metric("Price",f"${last['Close']:.2f}")
    col2.metric("RSI",f"{last['RSI']:.1f}")
    col3.metric("MACD",f"{last['MACD']:.2f}")
    col4.metric("ADX",f"{last['ADX']:.1f}")
    col5.metric("ATR",f"{last['ATR']:.2f}")
    st.markdown(f"### 🔹 **Signal: {signal}** (Score {score:.2f}, News {news:+.2f})")
    st.plotly_chart(plot_all(ind,ticker),use_container_width=True)
    st.markdown(explain(ind,fund,news,signal))
    st.progress(min(abs(score)/5,1.0))
    st.write(f"**Backtest 5-year RSI model:** {acc}% accuracy")

    with st.expander("📘 Learn Indicators & Strategy Education"):
        st.markdown("""
**RSI** — Relative Strength Index <30 = oversold, >70 = overbought  
**MACD** — Momentum oscillator tracking trend reversals  
**Bollinger Bands** — Volatility range; price outside bands signals extremes  
**ADX** — Trend strength >25 = strong trend  
**ATR** — Average True Range = volatility measure  
**P/E Ratio** — Valuation metric; lower can mean undervalued  
**Analyst Mean Rating** — 1=Strong Buy … 5=Strong Sell  
**Tip:** Combine technical oversold zones with positive sentiment for best entries.
""")

st.markdown("---")
st.caption("© 2025 Raj Gupta — AI Stock Signals Dashboard PRO v5.1 | Educational use only")
