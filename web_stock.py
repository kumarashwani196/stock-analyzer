import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- 1. वेबसाइट की सजावट (Page Config & CSS) ---
st.set_page_config(page_title="Pro Stock Analyzer", layout="wide", page_icon="💎")

# कस्टम रंग और फॉन्ट (HTML/CSS का इस्तेमाल)
st.markdown("""
<style>
.big-font { font-size:42px !important; color: #1E88E5; font-weight: bold; text-align: center; }
.sub-text { font-size:20px !important; color: #FF9800; text-align: center; margin-bottom: 30px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">💎 AI प्रो स्टॉक एनालाइज़र</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">स्मार्ट, तेज़ और सटीक शेयर बाज़ार डैशबोर्ड</p>', unsafe_allow_html=True)
st.markdown("---")

# --- 2. साइडबार (Sidebar - सर्च बॉक्स को साइड में करना) ---
st.sidebar.header("🔍 स्टॉक चुनें")
st.sidebar.write("यहाँ किसी भी अमेरिकी शेयर का नाम डालें:")
ticker_symbol = st.sidebar.text_input("Ticker (जैसे AAPL, NVDA, TSLA):", "AAPL").upper()

# --- 3. मेन काम शुरू (बटन दबाने पर) ---
if st.sidebar.button("एनालिसिस शुरू करें 🚀"):
    with st.spinner(f"{ticker_symbol} का डेटा स्कैन हो रहा है..."):
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="1y")
        info = stock.info
        
        if df.empty:
            st.error("डेटा नहीं मिला! कृपया सही नाम डालें।")
        else:
            current_price = df['Close'].iloc[-1]
            
            # बड़ा और सुंदर प्राइस टैग
            st.metric(label=f"💰 {ticker_symbol} का मौजूदा प्राइस (USD)", value=f"${current_price:.2f}")
            st.write("") # थोड़ी खाली जगह
            
            # --- 4. टैब्स (Tabs) बनाना (ताकि पेज प्रोफेशनल लगे) ---
            tab1, tab2, tab3 = st.tabs(["📊 चार्ट & ग्राफ", "🏢 फंडामेंटल & टेक्निकल", "🎯 AI फैसला"])
            
            # पहला टैब: सिर्फ ग्राफ
            with tab1:
                fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='#00E676', width=3))])
                fig.update_layout(title="पिछले 1 साल का प्राइस मूवमेंट", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            # दूसरा टैब: कंपनी की जानकारी
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.info("📊 कंपनी की सेहत (Fundamentals)")
                    st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**EPS (कमाई):** ${info.get('trailingEps', 'N/A')}")
                    st.write(f"**कर्ज़ा (Debt/Equity):** {info.get('debtToEquity', 'N/A')}")
                
                with col2:
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    delta = df['Close'].diff(1)
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    st.warning("⚙️ टेक्निकल इंडिकेटर्स")
                    st.write(f"**SMA (20 दिन):** ${df['SMA_20'].iloc[-1]:.2f}")
                    st.write(f"**RSI (स्ट्रेंथ):** {df['RSI'].iloc[-1]:.2f}")
            
            # तीसरा टैब: फाइनल रिजल्ट
            with tab3:
                st.subheader("💡 मशीन लर्निंग / AI प्रिडिक्शन")
                rsi_val = df['RSI'].iloc[-1]
                sma_val = df['SMA_20'].iloc[-1]
                
                if rsi_val < 40 and current_price > sma_val:
                    st.success("✅ **फाइनल फैसला: BUY (खरीदें)** - शेयर मज़बूत स्थिति में है और अच्छी कीमत पर मिल रहा है।")
                elif rsi_val > 65:
                    st.error("❌ **फाइनल फैसला: SELL (बेचें / बचें)** - शेयर अभी बहुत ज़्यादा महँगा (Overbought) हो गया है।")
                else:
                    st.warning("⚠️ **फाइनल फैसला: HOLD (इंतज़ार करें)** - अभी शेयर किसी भी दिशा में जा सकता है, इंतज़ार करना बेहतर है।")