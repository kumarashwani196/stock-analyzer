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
    out.append(("MACD", macd, "Bullish" if macd > macds else "Bearish"))

    stoch = latest_value(d, "Stoch_K")
    out.append(("Stochastic", stoch, "Bullish" if stoch < 20 else "Bearish" if stoch > 80 else "Neutral"))

    wr = latest_value(d, "Williams_R")
    out.append(("Williams %R", wr, "Bullish" if wr < -80 else "Bearish" if wr > -20 else "Neutral"))

    cci = latest_value(d, "CCI")
    out.append(("CCI", cci, "Bullish" if cci > 100 else "Bearish" if cci < -100 else "Neutral"))

    adx = latest_value(d, "ADX")
    dip = latest_value(d, "DI_Plus")
    dim = latest_value(d, "DI_Minus")
    adx_sig = "Bullish" if dip > dim and adx >= 20 else "Bearish" if dim > dip and adx >= 20 else "Neutral"
    out.append(("ADX / DI", adx, adx_sig))

    bb_u = latest_value(d, "BB_Upper")
    bb_l = latest_value(d, "BB_Lower")
    bb_sig = "Bullish" if c < bb_l else "Bearish" if c > bb_u else "Neutral"
    out.append(("Bollinger Bands", c, bb_sig))

    mfi = latest_value(d, "MFI")
    out.append(("MFI", mfi, "Bullish" if mfi < 20 else "Bearish" if mfi > 80 else "Neutral"))

    vwap = latest_value(d, "VWAP")
    out.append(("VWAP", vwap, "Bullish" if c > vwap else "Bearish"))

    obv = latest_value(d, "OBV")
    obv_ma = latest_value(d, "OBV_SMA20")
    out.append(("OBV", obv, "Bullish" if obv > obv_ma else "Bearish"))

    roc = latest_value(d, "ROC_12")
    out.append(("ROC", roc, "Bullish" if roc > 0 else "Bearish"))

    mom = latest_value(d, "Momentum_10")
    out.append(("Momentum", mom, "Bullish" if mom > 0 else "Bearish"))

    return out

# -------------------- Sidebar --------------------

st.sidebar.header("🔍 Stock Analyzer")
ticker_symbol = st.sidebar.text_input(
    "US Ticker",
    "AAPL",
    help="Examples: AAPL, NVDA, MSFT, TSLA, AMZN"
).strip().upper()

period = st.sidebar.selectbox(
    "Historical period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

if st.sidebar.button("🚀 ANALYZE STOCK", use_container_width=True):
    st.session_state["run_analysis"] = True

if st.session_state.get("run_analysis", False):
    with st.spinner(f"{ticker_symbol} का market data analyze हो रहा है..."):
        try:
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period=period, auto_adjust=False)
            info = stock.info

            if df.empty:
                st.error("डेटा नहीं मिला। सही US ticker डालें।")
                st.stop()

            df = add_indicators(df)
            current_price = latest_value(df, "Close")
            signals = technical_signals(df)

            bullish = sum(1 for x in signals if x[2] == "Bullish")
            bearish = sum(1 for x in signals if x[2] == "Bearish")
            neutral = sum(1 for x in signals if x[2] == "Neutral")

            st.metric(
                f"💰 {ticker_symbol} Current Price",
                f"${current_price:,.2f}"
            )

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Price & Indicators",
                "⚙️ Technical",
                "🏢 Fundamentals",
                "📰 News",
                "🌎 Global & Investors"
            ])

            # -------------------- Chart --------------------
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Price"
                ))
                for col in ["SMA_20", "SMA_50", "EMA_21", "EMA_200"]:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[col], mode="lines", name=col
                    ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["BB_Upper"], mode="lines",
                    name="BB Upper", line=dict(dash="dot")
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["BB_Lower"], mode="lines",
                    name="BB Lower", line=dict(dash="dot")
                ))
                fig.update_layout(
                    title=f"{ticker_symbol} Price + Moving Averages + Bollinger Bands",
                    height=650,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Bullish signals", bullish)
                c2.metric("Bearish signals", bearish)
                c3.metric("Neutral signals", neutral)

                st.caption("Signals are indicator readings, not guaranteed future outcomes.")

            # -------------------- Technical --------------------
            with tab2:
                st.subheader("📊 Technical Indicator Dashboard")

                rows = []
                for name, value, sig in signals:
                    rows.append({
                        "Indicator": name,
                        "Latest Value": (
                            round(value, 2) if pd.notna(value) else "N/A"
                        ),
                        "Signal": sig
                    })

                tech_df = pd.DataFrame(rows)
                st.dataframe(tech_df, use_container_width=True, hide_index=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Momentum")
                    st.write(f"RSI: **{latest_value(df, 'RSI'):.2f}**")
                    st.write(f"MACD: **{latest_value(df, 'MACD'):.4f}**")
                    st.write(f"Stochastic %K: **{latest_value(df, 'Stoch_K'):.2f}**")
                    st.write(f"CCI: **{latest_value(df, 'CCI'):.2f}**")
                    st.write(f"ADX: **{latest_value(df, 'ADX'):.2f}**")

                with col2:
                    st.subheader("Price Levels")
                    st.write(f"VWAP: **${latest_value(df, 'VWAP'):.2f}**")
                    st.write(f"Support (20): **${latest_value(df, 'Support_20'):.2f}**")
                    st.write(f"Resistance (20): **${latest_value(df, 'Resistance_20'):.2f}**")
                    st.write(f"Pivot: **${latest_value(df, 'Pivot'):.2f}**")
                    st.write(f"ATR: **${latest_value(df, 'ATR'):.2f}**")

                st.subheader("📌 MACD")
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(
                    x=df.index, y=df["MACD"], name="MACD", mode="lines"
                ))
                macd_fig.add_trace(go.Scatter(
                    x=df.index, y=df["MACD_Signal"], name="Signal", mode="lines"
                ))
                macd_fig.add_bar(
                    x=df.index, y=df["MACD_Hist"], name="Histogram"
                )
                macd_fig.update_layout(height=400)
                st.plotly_chart(macd_fig, use_container_width=True)

            # -------------------- Fundamentals --------------------
            with tab3:
                st.subheader("🏢 Company Fundamentals")

                fundamental_fields = {
                    "Company": "longName",
                    "Sector": "sector",
                    "Industry": "industry",
                    "Market Cap": "marketCap",
                    "P/E (TTM)": "trailingPE",
                    "Forward P/E": "forwardPE",
                    "EPS": "trailingEps",
                    "Price/Book": "priceToBook",
                    "ROE": "returnOnEquity",
                    "ROA": "returnOnAssets",
                    "Debt/Equity": "debtToEquity",
                    "Profit Margin": "profitMargins",
                    "Operating Margin": "operatingMargins",
                    "Revenue Growth": "revenueGrowth",
                    "Earnings Growth": "earningsGrowth",
                    "Free Cash Flow": "freeCashflow",
                    "Dividend Yield": "dividendYield"
                }

                f_rows = []
                for label, key in fundamental_fields.items():
                    value = info.get(key, "N/A")
                    if key in ["marketCap", "freeCashflow"] and isinstance(value, (int, float)):
                        value = f"${value:,.0f}"
                    elif key in ["returnOnEquity", "returnOnAssets", "profitMargins",
                                  "operatingMargins", "revenueGrowth", "earningsGrowth",
                                  "dividendYield"] and isinstance(value, (int, float)):
                        value = f"{value * 100:.2f}%"
                    f_rows.append({"Metric": label, "Value": value})

                st.dataframe(
                    pd.DataFrame(f_rows),
                    use_container_width=True,
                    hide_index=True
                )

            # -------------------- News --------------------
            with tab4:
                st.subheader("📰 Latest Available Company News")

                try:
                    news = stock.news
                except Exception:
                    news = []

                if news:
                    for item in news[:15]:
                        content = item.get("content", item)
                        title = content.get("title", item.get("title", "News"))
                        publisher = content.get("provider", {}).get(
                            "displayName",
                            item.get("publisher", "Unknown")
                        )
                        link = (
                            content.get("canonicalUrl", {}).get("url")
                            or content.get("clickThroughUrl", {}).get("url")
                            or item.get("link")
                        )
                        pub_time = content.get("pubDate", "")

                        st.markdown(f"### {title}")
                        st.caption(f"{publisher} • {pub_time}")
                        if link:
                            st.markdown(f"[Open source article]({link})")
                        st.divider()
                else:
                    st.info("इस ticker के लिए उपलब्ध news data नहीं मिला।")

                st.caption(
                    "News availability depends on the data provider. "
                    "The analyzer does not treat a headline alone as proof of future price direction."
                )

            # -------------------- Global + Investors --------------------
            with tab5:
                st.subheader("🌎 Global Market Snapshot")

                global_tickers = {
                    "S&P 500": "^GSPC",
                    "Nasdaq": "^IXIC",
                    "Dow Jones": "^DJI",
                    "VIX": "^VIX",
                    "US 10Y Yield": "^TNX",
                    "Gold": "GC=F",
                    "Crude Oil": "CL=F",
                    "USD Index": "DX-Y.NYB",
                    "Bitcoin": "BTC-USD"
                }

                global_rows = []
                for name, symbol in global_tickers.items():
                    try:
                        g = yf.Ticker(symbol).history(period="5d")
                        if not g.empty:
                            last = float(g["Close"].iloc[-1])
                            prev = float(g["Close"].iloc[-2]) if len(g) > 1 else np.nan
                            change = ((last / prev) - 1) * 100 if pd.notna(prev) and prev else np.nan
                            global_rows.append({
                                "Market": name,
                                "Value": round(last, 4),
                                "1D Change %": round(change, 2) if pd.notna(change) else "N/A"
                            })
                    except Exception:
                        pass

                if global_rows:
                    st.dataframe(
                        pd.DataFrame(global_rows),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Global market data temporarily unavailable.")

                st.subheader("🏦 Institutional / Insider Data")

                try:
                    major = stock.major_holders
                    if major is not None and not major.empty:
                        st.write("Major holders")
                        st.dataframe(major, use_container_width=True, hide_index=True)
                except Exception:
                    st.info("Major holder data unavailable.")

                try:
                    inst = stock.institutional_holders
                    if inst is not None and not inst.empty:
                        st.write("Institutional holders")
                        st.dataframe(inst, use_container_width=True, hide_index=True)
                except Exception:
                    st.info("Institutional holder data unavailable.")

                try:
                    insider = stock.insider_transactions
                    if insider is not None and not insider.empty:
                        st.write("Recent insider transactions")
                        st.dataframe(insider.head(25), use_container_width=True, hide_index=True)
                except Exception:
                    st.info("Insider transaction data unavailable.")

            # -------------------- Summary --------------------
            st.markdown("---")
            st.subheader("🧭 Transparent Analysis Summary")

            if bullish > bearish:
                technical_summary = "Technical indicators currently have more Bullish readings than Bearish readings."
            elif bearish > bullish:
                technical_summary = "Technical indicators currently have more Bearish readings than Bullish readings."
            else:
                technical_summary = "Technical indicators are currently mixed."

            st.info(technical_summary)
            st.write(
                "यह summary उपलब्ध market data पर आधारित है। "
                "यह guaranteed return, price prediction या व्यक्तिगत investment advice नहीं है।"
            )

        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.info("यदि error बना रहे तो ticker और error message भेजें।")
else:
    st.info("👈 Sidebar में ticker डालें और **ANALYZE STOCK** दबाएँ।")
    st.markdown("""
    ### अभी इस version में
    - Moving averages
    - RSI / MACD
    - Bollinger Bands
    - ATR / ADX
    - Stochastic / Williams %R
    - CCI / ROC / Momentum
    - OBV / MFI / VWAP
    - Support / Resistance / Pivot
    - Fundamentals
    - Available company news
    - Global market snapshot
    - Institutional / insider data
    """)
