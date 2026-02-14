# QuantEdge: Stock Prediction System

A professional AI-powered stock prediction and trading dashboard.

## Features

- **📈 Advanced Prediction**: Uses an MLP Neural Network to forecast next-day stock prices.
- **⚡ Trading Signals**: Generates Buy/Sell/Hold signals with confidence scores.
- **🛡️ Risk Management**: Tracks Volatility, Win Rate, and Max Drawdown.
- **🧠 Explanation**: Provides reasoning for every trading decision (e.g., "High Volatility").
- **📊 Interactive Dashboard**: Built with Streamlit and Plotly for real-time analytics.

## Setup Instructions

1.  **Install Python**: Ensure Python 3.8+ is installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App**:
    Double-click `run_app.bat` OR run:
    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Select Ticker**: Choose a stock symbol (e.g., AAPL, NVDA) from the sidebar.
2.  **Prediction Tab**: View the forecasted price and model accuracy.
3.  **Trading Signal Tab**: Execute trades using the Smart Order Entry form.
4.  **Agent Portfolio Tab**: Analyze your performance, run backtests, and view drawdown charts.

## Technologies

- Python
- Streamlit
- Scikit-Learn (MLP Regressor)
- Plotly (Interactive Charts)
- yfinance (Market Data)
