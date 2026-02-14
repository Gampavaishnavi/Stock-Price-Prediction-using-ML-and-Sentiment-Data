import pandas as pd
import numpy as np
import yfinance as yf

def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data using yfinance.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for {ticker}. Using Mock Data.")
            return generate_mock_stock_data(ticker, start_date, end_date)

        # Flatten MultiIndex columns if present (common in yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        return df
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return generate_mock_stock_data(ticker, start_date, end_date)

def generate_mock_stock_data(ticker, start_date, end_date):
    """
    Generates realistic looking random stock data for testing/demo purposes.
    """
    print(f"Generating mock data for {ticker}...")
    dates = pd.date_range(start=start_date, end=end_date)
    n = len(dates)
    
    # Random walk
    start_price = 150.0
    returns = np.random.normal(0.0005, 0.02, n) # Mean return 0.05%, std 2%
    price_multipliers = np.cumprod(1 + returns)
    prices = start_price * price_multipliers
    
    # Generate OHLC
    opens = prices * (1 + np.random.normal(0, 0.005, n))
    closes = prices
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volumes = np.random.randint(1000000, 5000000, n)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    return df

def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_indicators(df):
    """
    Adds technical indicators to the dataframe.
    """
    if df is None or df.empty:
        return None
    
    df = df.copy()
    
    # Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
    
    # RSI
    df['RSI'] = calculate_rsi(df)
    
    # Drop NaN values created by rolling windows
    df.dropna(inplace=True)
    
    return df
