import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# PRO STOCK ANALYZER - VERSION 2
# Technical + Fundamental + Global Market + News + Investor Data
# ============================================================

st.set_page_config(
    page_title="Pro Stock Analyzer",
    layout="wide",
    page_icon="💎"
)

st.markdown("""
<style>
.big-font {font-size:42px !important; font-weight:bold; text-align:center;}
.sub-text {font-size:18px !important; text-align:center; margin-bottom:20px;}
.signal-card {padding:12px; border-radius:10px; border:1px solid #ddd; margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">💎 Pro Stock Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-text">Technical • Fundamental • News • Global Market • Institutional Data</p>',
    unsafe_allow_html=True
)

# -------------------- Indicator functions --------------------

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def add_indicators(df):
    d = df.copy()

    # Moving averages
    for p in [20, 50, 100, 200]:
        d[f"SMA_{p}"] = d["Close"].rolling(p).mean()
    for p in [9, 21, 50, 200]:
        d[f"EMA_{p}"] = d["Close"].ewm(span=p, adjust=False).mean()

    # RSI
    d["RSI"] = rsi(d["Close"], 14)

    # MACD
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

    # Bollinger Bands
    bb_mid = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_Middle"] = bb_mid
    d["BB_Upper"] = bb_mid + 2 * bb_std
    d["BB_Lower"] = bb_mid - 2 * bb_std

    # ATR
    d["ATR"] = atr(d, 14)

    # ADX / DI
    up_move = d["High"].diff()
    down_move = -d["Low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=d.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=d.index
    )
    atr14 = d["ATR"].replace(0, np.nan)
    d["DI_Plus"] = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    d["DI_Minus"] = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx = 100 * (d["DI_Plus"] - d["DI_Minus"]).abs() / (
        d["DI_Plus"] + d["DI_Minus"]
    ).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Stochastic
    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch_K"] = 100 * (d["Close"] - low14) / (high14 - low14).replace(0, np.nan)
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    # Williams %R
    d["Williams_R"] = -100 * (high14 - d["Close"]) / (high14 - low14).replace(0, np.nan)

    # CCI
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    tp_sma = tp.rolling(20).mean()
    mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d["CCI"] = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))

    # ROC and Momentum
    d["ROC_12"] = d["Close"].pct_change(12) * 100
    d["Momentum_10"] = d["Close"] - d["Close"].shift(10)

    # OBV
    direction = np.sign(d["Close"].diff()).fillna(0)
    d["OBV"] = (direction * d["Volume"].fillna(0)).cumsum()
    d["OBV_SMA20"] = d["OBV"].rolling(20).mean()

    # MFI
    raw_mf = tp * d["Volume"].fillna(0)
    pos_mf = raw_mf.where(tp.diff() > 0, 0).rolling(14).sum()
    neg_mf = raw_mf.where(tp.diff() < 0, 0).rolling(14).sum().abs()
    money_ratio = pos_mf / neg_mf.replace(0, np.nan)
    d["MFI"] = 100 - (100 / (1 + money_ratio))

    # VWAP
    typical = (d["High"] + d["Low"] + d["Close"]) / 3
    cumulative_volume = d["Volume"].fillna(0).cumsum()
    d["VWAP"] = (typical * d["Volume"].fillna(0)).cumsum() / cumulative_volume.replace(0, np.nan)

    # Volume average
    d["Volume_SMA20"] = d["Volume"].rolling(20).mean()

    # Simple support/resistance using rolling range
    d["Support_20"] = d["Low"].rolling(20).min()
    d["Resistance_20"] = d["High"].rolling(20).max()

    # Pivot levels from previous daily candle
    prev_high = d["High"].shift(1)
    prev_low = d["Low"].shift(1)
    prev_close = d["Close"].shift(1)
    d["Pivot"] = (prev_high + prev_low + prev_close) / 3
    d["R1"] = 2 * d["Pivot"] - prev_low
    d["S1"] = 2 * d["Pivot"] - prev_high
    d["R2"] = d["Pivot"] + (prev_high - prev_low)
    d["S2"] = d["Pivot"] - (prev_high - prev_low)

    return d

def latest_value(df, col):
    try:
        value = df[col].iloc[-1]
        return float(value) if pd.notna(value) else np.nan
    except Exception:
        return np.nan

def signal(condition_up, condition_down):
    if condition_up:
        return "Bullish"
    if condition_down:
        return "Bearish"
    return "Neutral"

def technical_signals(d):
    c = latest_value(d, "Close")
    out = []

    checks = [
        ("SMA 20", latest_value(d, "SMA_20"), signal(c > latest_value(d, "SMA_20"), c < latest_value(d, "SMA_20"))),
        ("SMA 50", latest_value(d, "SMA_50"), signal(c > latest_value(d, "SMA_50"), c < latest_value(d, "SMA_50"))),
        ("SMA 100", latest_value(d, "SMA_100"), signal(c > latest_value(d, "SMA_100"), c < latest_value(d, "SMA_100"))),
        ("SMA 200", latest_value(d, "SMA_200"), signal(c > latest_value(d, "SMA_200"), c < latest_value(d, "SMA_200"))),
        ("EMA 9", latest_value(d, "EMA_9"), signal(c > latest_value(d, "EMA_9"), c < latest_value(d, "EMA_9"))),
        ("EMA 21", latest_value(d, "EMA_21"), signal(c > latest_value(d, "EMA_21"), c < latest_value(d, "EMA_21"))),
        ("EMA 50", latest_value(d, "EMA_50"), signal(c > latest_value(d, "EMA_50"), c < latest_value(d, "EMA_50"))),
        ("EMA 200", latest_value(d, "EMA_200"), signal(c > latest_value(d, "EMA_200"), c < latest_value(d, "EMA_200"))),
    ]

    for name, value, sig in checks:
        out.append((name, value, sig))

    r = latest_value(d, "RSI")
    out.append(("RSI", r, "Bullish" if r < 30 else "Bearish" if r > 70 else "Neutral"))

    macd = latest_value(d, "MACD")
    macds = latest_value(d, "MACD_Signal")
    out…
