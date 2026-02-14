import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib
import os
import yfinance as yf
from datetime import date, timedelta
from utils.data_loader import download_data, add_technical_indicators
from utils.trading import get_trading_signal
from sentiment.analyzer import SentimentAnalyzer
from agent.simple_agent import TradingAgent
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
st.set_page_config(layout="wide", page_title="QuantEdge: Stock Prediction")
MODEL_PATH = "models/mlp_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# --- Session State for Agent ---
if 'agent' not in st.session_state:
    st.session_state.agent = TradingAgent(initial_balance=10000)

# --- Load Model (Lazy Loading) ---
@st.cache_resource
def load_models():
    scaler = None
    model = None
    model_available = False
    
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    
    # Try loading MLP Scikit-Learn model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            model_available = True
    except Exception as e:
        st.error(f"Error loading model: {e}")

    return model, scaler, model_available

# --- Sidebar ---
# --- Sidebar ---
try:
    st.sidebar.header("User Input")
    
    # --- BROKER INTEGRATION (Paper Trading) ---
    with st.sidebar.expander("🔌 Connect Broker (Paper Trading)"):
        alpaca_key = st.text_input("Alpaca Key ID", type="password")
        alpaca_secret = st.text_input("Alpaca Secret Key", type="password")
        connect_broker = st.button("Connect to Alpaca")
        
        if connect_broker:
            if alpaca_key and alpaca_secret:
                try:
                    from agent.broker import AlpacaBroker
                    st.session_state['broker'] = AlpacaBroker(alpaca_key, alpaca_secret)
                    st.success("Connected to Alpaca Paper Trading!")
                    st.session_state['simulation_mode'] = False
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
            else:
                st.warning("Please enter both Key ID and Secret.")
    
    # Check Broker Status
    broker = st.session_state.get('broker', None)
    if broker:
        st.sidebar.success("🟢 Broker Connected")
        mode = st.sidebar.radio("Trading Mode", ["Paper Trading", "Simulation"])
        st.session_state['simulation_mode'] = (mode == "Simulation")
    else:
        st.session_state['simulation_mode'] = True
    
    # Popular Tickers List
    POPULAR_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
        "AMD", "INTC", "IBM", "ORCL", "CRM", "ADBE", "PYPL", "SQ", 
        "SHOP", "SPOT", "UBER", "ABNB", "JPM", "BAC", "V", "MA", "DIS",
        "Custom"
    ]
    
    # Main Ticker Selection
    ticker_select = st.sidebar.selectbox("Select Stock Symbol", POPULAR_TICKERS, index=0)
    if ticker_select == "Custom":
        ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
    else:
        ticker = ticker_select
        
    start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2023, 1, 1))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comparison")
    
    # Comparison Toggle
    enable_comparison = st.sidebar.checkbox("Compare with another stock?")
    compare_ticker = ""
    
    if enable_comparison:
        # Comparison Ticker Selection
        compare_select = st.sidebar.selectbox("Compare with Symbol", POPULAR_TICKERS, index=0)
        if compare_select == "Custom":
            compare_ticker = st.sidebar.text_input("Enter Compare Symbol", "")
        else:
            compare_ticker = compare_select
    else:
        compare_ticker = ""

except Exception as e:
    st.sidebar.error(f"Sidebar Error: {e}")
    ticker = "AAPL" # Default fallback
    start_date = date(2020, 1, 1)
    end_date = date(2023, 1, 1)

# --- Main Application Logic ---
try:
    # --- NEWS SENTIMENT INTEGRATION ---
    with st.sidebar.expander("📰 News Settings (Real-Time)"):
        news_api_key = st.text_input("NewsAPI Key (Optional)", type="password")
        if news_api_key:
            os.environ["NEWS_API_KEY"] = news_api_key
    
    # --- Main Page ---
    st.title("📈 QuantEdge: Stock Prediction System")

    # Function to load data
    @st.cache_data
    def load_data(ticker, start, end):
        # Fetch extra data for technical indicators and model context (warm-up period)
        start_buffer = start - timedelta(days=100)
        df = download_data(ticker, start_buffer, end)
        df = add_technical_indicators(df)
        
        # Integrate Sentiment Data (Simulated for consistency with Training)
        if df is not None and not df.empty:
            # Determine date range from actual data to match
            data_start = df['Date'].min()
            data_end = df['Date'].max()
            
            analyzer = SentimentAnalyzer()
            # We simulate sentiment for the whole range covering the data
            sentiment_df = analyzer.simulate_daily_sentiment(data_start, data_end)
            
            # Merge
            # Ensure compatible types
            df['Date'] = pd.to_datetime(df['Date'])
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            
            df = pd.merge(df, sentiment_df, on='Date', how='left')
            df['Sentiment'].fillna(0, inplace=True) # Fallback to neutral
            
        return df

    data_load_state = st.text('Loading data...')
    try:
        df = load_data(ticker, start_date, end_date)
        data_load_state.text('Loading data... done!')
    except Exception as e:
        data_load_state.error(f"Error loading data: {e}")
        st.stop()

    # Load comparison data if selected
    df_compare = None
    if compare_ticker:
        st.text(f'Loading comparison data for {compare_ticker}...')
        try:
            df_compare = load_data(compare_ticker, start_date, end_date)
        except Exception:
            st.warning(f"Could not load data for comparison ticker: {compare_ticker}")

    # --- LIVE SENTIMENT FETCH ---
    analyzer = SentimentAnalyzer()
    live_sentiment, headlines = analyzer.fetch_news_sentiment(ticker)
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Prediction", "Trading Signal", "Model Performance", "Agent Portfolio"])

    with tab1:
        st.subheader(f"{ticker} Stock Overview")
        
        # Display Live Sentiment
        s_col1, s_col2 = st.columns([1, 2])
        with s_col1:
            sentiment_color = "green" if live_sentiment > 0.05 else "red" if live_sentiment < -0.05 else "gray"
            st.markdown(f"### Live Sentiment")
            st.markdown(f"<h2 style='color: {sentiment_color}'>{live_sentiment:.2f}</h2>", unsafe_allow_html=True)
            if live_sentiment > 0.05:
                st.write("Market is **Bullish** 🐂")
            elif live_sentiment < -0.05:
                st.write("Market is **Bearish** 🐻")
            else:
                st.write("Market is **Neutral** 😐")
        
        with s_col2:
            with st.expander("📰 Latest Headlines (NewsAPI / FinViz)", expanded=False):
                if headlines:
                    for h in headlines[:5]:
                        st.write(f"• {h}")
                else:
                    st.info("No headlines found.")
        
        st.markdown("---")
        if df is not None:
            # Filter data for display logic ONLY
            # This allows the model (Tab 2) to use the full buffered history
            df_display = df.copy()
            if 'Date' in df_display.columns:
                df_display['Date'] = pd.to_datetime(df_display['Date'])
                mask = (df_display['Date'].dt.date >= start_date) & (df_display['Date'].dt.date <= end_date)
                df_display = df_display.loc[mask]
                
            # Candlestick Chart (with optional BB)
            fig = go.Figure()
            
            # 1. Main Price & Bollinger Bands
            fig.add_trace(go.Candlestick(x=df_display['Date'],
                            open=df_display['Open'],
                            high=df_display['High'],
                            low=df_display['Low'],
                            close=df_display['Close'],
                            name=ticker))
            
            # Bollinger Bands Toggle
            show_bb = st.checkbox("Show Bollinger Bands", value=False)
            if show_bb and 'BB_Upper' in df_display.columns:
                fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['BB_Upper'], 
                                         line=dict(color='gray', width=1), name='Upper BB', showlegend=False))
                fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['BB_Lower'], 
                                         line=dict(color='gray', width=1), name='Lower BB', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            
            # Comparison Stock (Line Chart)
            if df_compare is not None and not df_compare.empty:
                df_comp_display = df_compare.copy()
                if 'Date' in df_comp_display.columns:
                    df_comp_display['Date'] = pd.to_datetime(df_comp_display['Date'])
                    mask_comp = (df_comp_display['Date'].dt.date >= start_date) & (df_comp_display['Date'].dt.date <= end_date)
                    df_comp_display = df_comp_display.loc[mask_comp]
                    
                fig.add_trace(go.Scatter(x=df_comp_display['Date'], y=df_comp_display['Close'], 
                                         mode='lines', name=f'{compare_ticker} Close', 
                                         line=dict(color='orange', width=2)))
                
            fig.update_layout(title=f'{ticker} Stock Price', xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. MACD Chart
            if 'MACD' in df_display.columns:
                st.subheader("MACD Oscillator")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df_display['Date'], y=df_display['MACD'], name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df_display['Date'], y=df_display['Signal_Line'], name='Signal', line=dict(color='orange')))
                
                # Color histogram based on positive/negative
                colors = np.where(df_display['MACD'] - df_display['Signal_Line'] > 0, 'green', 'red')
                fig_macd.add_trace(go.Bar(x=df_display['Date'], y=df_display['MACD'] - df_display['Signal_Line'], name='Histogram', marker_color=colors))
                
                fig_macd.update_layout(height=300, xaxis_title="Date", yaxis_title="MACD", showlegend=True)
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Raw Data
            if st.checkbox("Show Raw Data"):
                st.write(df_display.tail())
        else:
            st.error("No data found for this ticker.")

    with tab2:
        st.subheader("Price Prediction (Next Day)")
        
        model, scaler, model_available = load_models()
        
        if df is not None and len(df) > 60:
            # Prepare latest sequence
            # Features: Close, MA_10, MA_50, Volatility, RSI, Sentiment
            
            # We need the last 60 days of data for features
            feature_cols = ['Close', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'Sentiment']
            
            # Match features with training
            df_latest = df.tail(60).copy()
            
            if scaler and model_available and model:
                # Scale
                try:
                    # Ensure columns exist
                    missing_cols = [c for c in feature_cols if c not in df_latest.columns]
                    if missing_cols:
                       st.warning(f"Missing features for prediction: {missing_cols}")
                    else:
                        input_data = df_latest[feature_cols].values
                        scaled_input = scaler.transform(input_data)
                        
                        # Flatten for MLP: (1, 60*features)
                        X_input = scaled_input.flatten().reshape(1, -1)
                        
                        # Predict
                        predicted_scaled = model.predict(X_input)
                        
                        # Inverse transform
                        # We predicted the scaled value of 'Close' (index 0)
                        # Create a dummy row to inverse
                        dummy_row = np.zeros((1, len(feature_cols)))
                        dummy_row[0, 0] = predicted_scaled[0]
                        predicted_price = scaler.inverse_transform(dummy_row)[0, 0]
                        
                        current_price = df['Close'].iloc[-1]
                        
                        st.metric(label="Predicted Next Close (ANN/MLP)", value=f"${predicted_price:.2f}", delta=f"${predicted_price - current_price:.2f}")
                        
                        # --- Feature Visualization ---
                        st.markdown("### 📊 Model Input Features")
                        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                        f_col1.metric("RSI (14)", f"{df_latest['RSI'].iloc[-1]:.2f}")
                        f_col2.metric("Sentiment", f"{df_latest['Sentiment'].iloc[-1]:.2f}")
                        f_col3.metric("MA (10)", f"${df_latest['MA_10'].iloc[-1]:.2f}")
                        f_col4.metric("Volatility", f"{df_latest['Volatility'].iloc[-1]:.4f}")

                        # --- Real-Time News Analysis (New Feature) ---
                        st.markdown("---")
                        st.subheader("📰 Real-Time News Analysis")
                        
                        analyzer = SentimentAnalyzer()
                        with st.spinner(f"Fetching latest news for {ticker}..."):
                            news_score, headlines = analyzer.fetch_news_sentiment(ticker)
                        
                        n_col1, n_col2 = st.columns([1, 2])
                        with n_col1:
                            st.metric("Live News Sentiment", f"{news_score:.2f}", 
                                      delta="Bullish" if news_score > 0.05 else "Bearish" if news_score < -0.05 else "Neutral")
                            
                            if news_score > 0.05:
                                st.success("Market sentiment is **Bullish** based on recent news.")
                            elif news_score < -0.05:
                                st.error("Market sentiment is **Bearish** based on recent news.")
                            else:
                                st.info("Market sentiment is **Neutral**.")
                                
                        with n_col2:
                            st.markdown("**Top Headlines:**")
                            for h in headlines[:5]:
                                st.markdown(f"- {h}")
                        st.markdown("---")
                        
                        # --- Recent Accuracy (Backtest on view) ---
                        st.markdown("### 🔍 Recent Model Accuracy (Last 30 Days)")
                        
                        if len(df) > 90:
                            # Create batch for last 30 days
                            # We need sequences for the last 30 days. Each sequence needs 60 days prior.
                            recent_data = df.tail(90).reset_index(drop=True)
                            
                            # Prepare batch inputs
                            X_batch = []
                            actuals = []
                            dates_batch = []
                            
                            # We want to predict for the last 30 days indices: 60 to 89
                            for i in range(60, len(recent_data)):
                                # Features for this day's prediction (using previous 60 days)
                                # Actually, to predict for index i, we need data from i-60 to i-1 ??? 
                                # No, standard LSTM/MLP usually takes T-60 to T-1 to predict T.
                                # So to predict recent_data[i], we use recent_data[i-60:i]
                                
                                feat_window = recent_data.iloc[i-60:i][feature_cols].values
                                if len(feat_window) == 60:
                                    scaled_window = scaler.transform(feat_window)
                                    X_batch.append(scaled_window.flatten())
                                    actuals.append(recent_data.iloc[i]['Close'])
                                    dates_batch.append(recent_data.iloc[i]['Date'])
                            
                            if X_batch:
                                X_batch = np.array(X_batch)
                                preds_scaled = model.predict(X_batch)
                                
                                # Inverse transform predictions
                                preds_inverse = []
                                for p in preds_scaled:
                                    dummy = np.zeros((1, len(feature_cols)))
                                    dummy[0, 0] = p
                                    val = scaler.inverse_transform(dummy)[0, 0]
                                    preds_inverse.append(val)
                                    
                                # Plot
                                acc_df = pd.DataFrame({
                                    'Date': dates_batch,
                                    'Actual': actuals,
                                    'Predicted': preds_inverse
                                })
                                
                                fig_acc = go.Figure()
                                fig_acc.add_trace(go.Scatter(x=acc_df['Date'], y=acc_df['Actual'], name='Actual', line=dict(color='blue')))
                                fig_acc.add_trace(go.Scatter(x=acc_df['Date'], y=acc_df['Predicted'], name='Predicted', line=dict(color='orange', dash='dash')))
                                fig_acc.update_layout(title="Actual vs Predicted (Last 30 Days)", xaxis_title="Date", yaxis_title="Price")
                                st.plotly_chart(fig_acc, use_container_width=True)
                                
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    
            else:
                st.warning("Model not found. Please run `train_model.py`.")
        else:
            st.warning("Not enough data to make predictions (Need > 60 days).")

    with tab3:
        st.subheader("Trading Signal")
        
        if 'predicted_price' in locals():
            current_price = df['Close'].iloc[-1]
            signal_data = get_trading_signal(current_price, predicted_price)
            
            action = signal_data['action']
            reason = signal_data['reason']
            confidence = signal_data['confidence']
            tp = signal_data['take_profit']
            sl = signal_data['stop_loss']
            
            col_sig1, col_sig2 = st.columns([1, 2])
            
            with col_sig1:
                color = "green" if action == "BUY" else "red" if action == "SELL" else "gray"
                # Display Signal with Non-Selectable CSS
                st.markdown(f"""
                <div style="text-align: center; user-select: none; cursor: default;">
                    <h1 style="color: {color}; margin: 0;">{action}</h1>
                    <p style="color: gray; margin-top: 5px;">Reason: {reason}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_sig2:
                # Gauge Chart for Confidence
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence (%)"},
                    gauge = {'axis': {'range': [0, 100]},
                             'bar': {'color': "darkblue"},
                             'steps' : [
                                 {'range': [0, 50], 'color': "lightgray"},
                                 {'range': [50, 80], 'color': "gray"}]}))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown("### ⚡ Execute Trade")
                
                # Determine Mode & Data
                sim_mode = st.session_state.get('simulation_mode', True)
                broker = st.session_state.get('broker', None)
                
                current_bal = 0.0
                current_pos = 0
                mode_label = "Simulation"
                
                if sim_mode:
                    current_bal = st.session_state.agent.balance
                    current_pos = st.session_state.agent.holdings
                    mode_label = "Simulation"
                else:
                    try:
                        current_bal = float(broker.get_balance())
                        # Need to find position for this specific ticker
                        positions = broker.get_positions()
                        for p in positions:
                            if p.symbol == ticker:
                                current_pos = int(p.qty)
                                break
                        mode_label = "Paper Trading"
                    except:
                        current_bal = 0.0
                        current_pos = 0
                        mode_label = "Error (Check Broker)"
                
                st.write(f"**Mode:** {mode_label}")
                st.write(f"**Available Cash:** ${current_bal:,.2f}")
                st.write(f"**Holdings:** {current_pos} shares")
                
                # Message Container
                msg_container = st.empty()
                
                # --- New Order Entry Form ---
                st.markdown("### 📝 Order Entry")
                
                with st.form("order_form"):
                    col_ord1, col_ord2 = st.columns(2)
                    
                    with col_ord1:
                        # Max shares affordable
                        max_buy = int(current_bal // current_price) if current_price > 0 else 0
                        qty = st.number_input("Quantity", min_value=1, max_value=10000, value=1, step=1)
                    
                    with col_ord2:
                        st.metric("Estimated Cost", f"${(qty * current_price):.2f}")
                    
                    c1, c2 = st.columns(2)
                    
                    # Buy Button
                    submitted_buy = c1.form_submit_button("🟢 BUY", use_container_width=True)
                    # Sell Button
                    submitted_sell = c2.form_submit_button("🔴 SELL", use_container_width=True)
                    
                    if submitted_buy:
                        cost = qty * current_price
                        if current_bal >= cost:
                            # Fetch current volatility
                            vol = df.iloc[-1]['Volatility'] if 'Volatility' in df.columns else 0.0
                            # Use Live Sentiment
                            sent = live_sentiment
                            
                            if sim_mode:
                                st.session_state.agent.execute_trade(df.iloc[-1]['Date'], "BUY", current_price, reason, quantity=qty, volatility=vol, sentiment=sent)
                                msg_container.success(f"✅ SIM: BOUGHT {qty} shares of {ticker} at ${current_price:.2f}!")
                            else:
                                order = broker.submit_order(ticker, qty, 'buy')
                                if order:
                                    msg_container.success(f"🚀 PAPER: Order Sent to Buy {qty} shares of {ticker}!")
                                else:
                                    msg_container.error("Order Failed to Submit.")
                            st.rerun()
                        else:
                            msg_container.error(f"Insufficient Funds! Max you can buy is {max_buy}.")
                            
                    if submitted_sell:
                        if current_pos >= qty:
                            # Fetch current volatility
                            vol = df.iloc[-1]['Volatility'] if 'Volatility' in df.columns else 0.0
                            # Use Live Sentiment
                            sent = live_sentiment
                            
                            if sim_mode:
                                st.session_state.agent.execute_trade(df.iloc[-1]['Date'], "SELL", current_price, reason, quantity=qty, volatility=vol, sentiment=sent)
                                msg_container.success(f"✅ SIM: SOLD {qty} shares of {ticker} at ${current_price:.2f}!")
                            else:
                                order = broker.submit_order(ticker, qty, 'sell')
                                if order:
                                    msg_container.success(f"🚀 PAPER: Order Sent to Sell {qty} shares!")
                                else:
                                    msg_container.error("Order Failed to Submit.")
                            st.rerun()
                        else:
                            msg_container.error(f"Insufficient Holdings! You only have {current_pos} shares.")
                
                # Show tiny portfolio summary if trades exist (Sim only for now in this view, expanded in dashboard)
                if sim_mode and st.session_state.agent.trade_history:
                    st.markdown("---")
                    st.write("### 💼 Portfolio Update")
                    p_col1, p_col2 = st.columns(2)
                    p_col1.metric("Cash Balance", f"${st.session_state.agent.balance:,.2f}")
                    p_col2.metric("Total Holdings", f"{st.session_state.agent.holdings} Shares")
                    
                    with st.expander("Recent Trades"):
                        st.dataframe(pd.DataFrame(st.session_state.agent.trade_history).tail(5))

            st.markdown("### 🛡️ Risk Management")
            rm_col1, rm_col2, rm_col3 = st.columns(3)
            rm_col1.metric("Entry Price", f"${current_price:.2f}")
            rm_col2.metric("Target (Take Profit)", f"${tp:.2f}", delta=f"{((tp-current_price)/current_price)*100:.2f}%")
            rm_col3.metric("Stop Loss", f"${sl:.2f}", delta=f"{((sl-current_price)/current_price)*100:.2f}%", delta_color="inverse")
        else:
             st.write("Calculate prediction first.")

    with tab4:
        st.subheader("Model Performance")
        model, scaler, model_available = load_models()
        
        if model_available:
            
            # --- 1. Real-time Metrics Calculation ---
            if df is not None and len(df) > 100:
                
                # Prepare data for all available points
                # We need 60 days lookback for each prediction
                feature_cols = ['Close', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'Sentiment']
                
                # Create dataset
                X_eval = []
                y_eval_actual = []
                dates_eval = []
                
                # Evaluate on the last 200 days (or fewer if data is short)
                eval_window = min(len(df) - 61, 365) 
                start_idx = len(df) - eval_window
                
                data_subset = df.iloc[start_idx-60:].reset_index(drop=True)
                
                for i in range(60, len(data_subset)):
                    feat_window = data_subset.iloc[i-60:i][feature_cols].values
                    target = data_subset.iloc[i]['Close']
                    
                    if len(feat_window) == 60:
                         # Scale
                         scaled_window = scaler.transform(feat_window)
                         X_eval.append(scaled_window.flatten())
                         y_eval_actual.append(target)
                         dates_eval.append(data_subset.iloc[i]['Date'])
                
                if X_eval:
                    X_eval = np.array(X_eval)
                    y_eval_actual = np.array(y_eval_actual)
                    
                    # Predict
                    y_pred_scaled = model.predict(X_eval)
                    
                    # Inverse Transform Predictions
                    y_pred = []
                    for p in y_pred_scaled:
                        dummy = np.zeros((1, len(feature_cols)))
                        dummy[0, 0] = p
                        val = scaler.inverse_transform(dummy)[0, 0]
                        y_pred.append(val)
                    y_pred = np.array(y_pred)
                    
                    # Metrics
                    r2 = r2_score(y_eval_actual, y_pred)
                    mae = mean_absolute_error(y_eval_actual, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_eval_actual, y_pred))
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{r2:.4f}")
                    col2.metric("MAE", f"${mae:.2f}")
                    col3.metric("RMSE", f"${rmse:.2f}")
                    
                    # --- 2. Visualizations ---
                    
                    # Predicted vs Actual Time Series
                    st.markdown("### 📉 Predicted vs Actual (Time Series)")
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=dates_eval, y=y_eval_actual, name="Actual", line=dict(color='blue')))
                    fig_ts.add_trace(go.Scatter(x=dates_eval, y=y_pred, name="Predicted", line=dict(color='orange', dash='dot')))
                    fig_ts.update_layout(xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                    # Scatter Plot (Correlation)
                    col_v1, col_v2 = st.columns(2)
                    
                    with col_v1:
                        st.markdown("### 🔗 Correlation Analysis")
                        fig_corr = go.Figure(data=go.Scatter(x=y_eval_actual, y=y_pred, mode='markers', marker=dict(color='purple', opacity=0.5)))
                        fig_corr.add_shape(type="line", x0=min(y_eval_actual), y0=min(y_eval_actual), x1=max(y_eval_actual), y1=max(y_eval_actual), line=dict(color="red", dash="dash"))
                        fig_corr.update_layout(xaxis_title="Actual Price", yaxis_title="Predicted Price", title="Actual vs Predicted")
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                    with col_v2:
                        st.markdown("### 📊 Residuals (Error Distribution)")
                        residuals = y_eval_actual - y_pred
                        fig_res = go.Figure(data=[go.Histogram(x=residuals, nbinsx=30, marker_color='teal')])
                        fig_res.update_layout(title="Prediction Error Distribution", xaxis_title="Error ($)", yaxis_title="Count")
                        st.plotly_chart(fig_res, use_container_width=True)

                    # --- 3. Loss Curve (if available) ---
                    if hasattr(model, 'loss_curve_'):
                        st.markdown("### 📉 Training Loss Curve")
                        fig_loss = go.Figure(data=go.Scatter(y=model.loss_curve_, mode='lines', line=dict(color='red')))
                        fig_loss.update_layout(title="Loss over Iterations", xaxis_title="Iteration", yaxis_title="Loss")
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
            else:
                st.info("Not enough data loaded to evaluate model performance.")
                
        else:
            st.warning("Model not trained yet. Run `train_model.py`.")

    with tab5:
        st.subheader("📊 Live Portfolio Dashboard")
        
        # Determine Mode
        sim_mode = st.session_state.get('simulation_mode', True)
        broker = st.session_state.get('broker', None)
        
        # --- Live Metrics ---
        if sim_mode:
            current_price = df['Close'].iloc[-1]
            portfolio_value = st.session_state.agent.balance + (st.session_state.agent.holdings * current_price)
            profit_loss = portfolio_value - st.session_state.agent.initial_balance
            roi = (profit_loss / st.session_state.agent.initial_balance) * 100
            cash_balance = st.session_state.agent.balance
            total_trades = len(st.session_state.agent.trade_history)
        else:
            # Broker Data
            try:
                portfolio_value = float(broker.get_equity())
                cash_balance = float(broker.get_balance())
                # For P/L, we can't easily know initial balance unless hardcoded or fetched from history API (complex)
                # We'll assume 100k for Paper Trading default, or just show daily P/L if available (Alpaca has it)
                # usage: account.equity - account.last_equity ? 
                # For now, just show current equity as primary metric.
                profit_loss = 0.0 # Placeholder
                roi = 0.0
                total_trades = "N/A"
            except:
                portfolio_value = 0.0
                cash_balance = 0.0
                profit_loss = 0.0
                roi = 0.0
                total_trades = "Err"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Worth", f"${portfolio_value:,.2f}")
        m2.metric("Cash Balance", f"${cash_balance:,.2f}")
        if sim_mode:
            m3.metric("Total Profit/Loss", f"${profit_loss:,.2f}", f"{roi:.2f}%")
            m4.metric("Total Trades", total_trades)
        else:
            m3.metric("Buying Power", f"${float(broker.api.get_account().buying_power):,.2f}" if broker else "$0")
            m4.metric("Status", broker.account.status if broker and hasattr(broker, 'account') else "Disconnected")

        # --- Visuals ---
        col_viz1, col_viz2 = st.columns(2)
        
        if sim_mode:
            with col_viz1:
                st.markdown("### 🥧 Asset Allocation")
                # Pie Chart: Cash vs Stock
                stock_value = st.session_state.agent.holdings * current_price
                cash_value = st.session_state.agent.balance
                
                fig_alloc = go.Figure(data=[go.Pie(labels=['Cash', 'Stock'], values=[cash_value, stock_value], hole=.3)])
                fig_alloc.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_alloc, use_container_width=True)
                
            with col_viz2:
                st.markdown("### 📈 Portfolio History")
                # History Line Chart
                history_df = st.session_state.agent.get_portfolio_history()
                if not history_df.empty:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(x=history_df['Date'], y=history_df['Portfolio_Value'], mode='lines', name='Net Worth', fill='tozeroy'))
                    fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Date", yaxis_title="Value ($)")
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No trading history yet. Make some trades!")
        else:
            # Broker Mode Visuals
            st.markdown("### 📋 Open Positions (Alpaca)")
            try:
                positions = broker.get_positions()
                if positions:
                    # Convert to DataFrame for display
                    pos_data = []
                    for p in positions:
                        pos_data.append({
                            "Symbol": p.symbol,
                            "Qty": p.qty,
                            "Entry Price": f"${float(p.avg_entry_price):.2f}",
                            "Current Price": f"${float(p.current_price):.2f}",
                            "Market Value": f"${float(p.market_value):.2f}",
                            "P/L": f"${float(p.unrealized_pl):.2f}"
                        })
                    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                else:
                    st.info("No open positions.")
            except Exception as e:
                st.error(f"Error fetching positions: {e}")

        # --- Live Trade Log ---
        st.markdown("### 📜 Trade Log")
        trade_df = st.session_state.agent.get_trade_history()
        if not trade_df.empty:
            st.dataframe(trade_df, use_container_width=True)
        
        st.markdown("---")
        
        # --- Strategy Sandbox (Backtest) ---
        with st.expander("🛠️ Strategy Sandbox (Backtest Simulation)"):
            st.write("Test the automated strategy on historical data.")
            # Run Simulation
            if st.button("Run Backtest Simulation"):
                with st.spinner("Running agent backtest..."):
                    from agent.simple_agent import TradingAgent
                    agent = TradingAgent(initial_balance=10000)
                    
                    if df is not None and len(df) > 60:
                        backtest_data = df.tail(200).copy().reset_index(drop=True) # Last 200 days
                        progress_bar = st.progress(0)
                        
                        for i in range(len(backtest_data) - 1):
                            current_date = backtest_data.loc[i, 'Date']
                            current_price = backtest_data.loc[i, 'Close']
                            
                            # Fetch Volatility and Sentiment for Backtest
                            vol = backtest_data.loc[i, 'Volatility'] if 'Volatility' in backtest_data.columns else 0.0
                            sent = backtest_data.loc[i, 'Sentiment'] if 'Sentiment' in backtest_data.columns else 0.0
                            
                            if backtest_data.loc[i, 'MA_10'] > backtest_data.loc[i, 'MA_50'] and backtest_data.loc[i, 'RSI'] < 70:
                                predicted_next = current_price * 1.01 
                            elif backtest_data.loc[i, 'MA_10'] < backtest_data.loc[i, 'MA_50'] and backtest_data.loc[i, 'RSI'] > 30:
                                predicted_next = current_price * 0.99 
                            else:
                                predicted_next = current_price 
                                
                            signal_dict = get_trading_signal(current_price, predicted_next)
                            action = signal_dict['action']
                            reason = signal_dict['reason']
                            
                            # Execute with full context
                            agent.execute_trade(current_date, action, current_price, reason, volatility=vol, sentiment=sent)
                            progress_bar.progress((i + 1) / len(backtest_data))
                            
                        st.success("Backtest Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        final_value = agent.portfolio_value
                        profit = final_value - agent.initial_balance
                        return_pct = (profit / agent.initial_balance) * 100
                        
                        # Calculate Win Rate
                        wins = len([t for t in agent.trade_history if t['Value'] > 0 and t['Action'] == 'SELL' and t.get('Reason', '').find('Profit') != -1]) 
                        
                        # Calculate Executed vs Skipped
                        # Note: Simple agent doesn't store Skipped trades in history by default unless we modified it to do so.
                        # Since we only modified execute_trade to RETURN if skipped, it might not be in the list.
                        # Let's check agent implementation previously.
                        # Actually, looking at agent/simple_agent.py (from memory/context), we log SKIPs now!
                        
                        skipped = len([t for t in agent.trade_history if t['Action'] == 'SKIP'])
                        executed = len([t for t in agent.trade_history if t['Action'] in ['BUY', 'SELL']])
                        
                        col1.metric("Final Value", f"${final_value:,.2f}")
                        col2.metric("Total P/L", f"${profit:,.2f}", f"{return_pct:.2f}%")
                        col3.metric("Trades Executed", executed)
                        col4.metric("Trades Skipped", skipped)
                        
                        history_df = agent.get_portfolio_history()
                        if not history_df.empty:
                            # 1. Portfolio Value Chart
                            st.markdown("#### 📈 Portfolio Growth")
                            fig_port = go.Figure()
                            fig_port.add_trace(go.Scatter(x=history_df['Date'], y=history_df['Portfolio_Value'], mode='lines', name='Portfolio Value'))
                            fig_port.update_layout(xaxis_title="Date", yaxis_title="Value ($)")
                            st.plotly_chart(fig_port, use_container_width=True)
                            
                            # 2. Drawdown Chart
                            if 'Drawdown' in history_df.columns:
                                st.markdown("#### 📉 Drawdown Risk")
                                fig_dd = go.Figure()
                                fig_dd.add_trace(go.Scatter(x=history_df['Date'], y=history_df['Drawdown'] * -100, mode='lines', name='Drawdown %', fill='tozeroy', line=dict(color='red')))
                                fig_dd.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
                                st.plotly_chart(fig_dd, use_container_width=True)
                        
                        st.subheader("Detailed Trade History (with Reasoning)")
                        trade_df = agent.get_trade_history()
                        if not trade_df.empty:
                            st.dataframe(trade_df, use_container_width=True)
                        else:
                            st.info("No trades executed.")
                    else:
                        st.warning("Not enough data for backtest.")
except Exception as e:
    st.error(f"Application Error: {e}")
