from utils.data_loader import download_data, add_technical_indicators
import pandas as pd

def test_data_loader():
    ticker = 'AAPL'
    print(f"Testing data loader for {ticker}...")
    df = download_data(ticker, '2023-01-01', '2024-01-01')
    
    if df is not None and not df.empty:
        print("Data downloaded successfully.")
        print(f"Shape: {df.shape}")
        print(df.head())
        
        print("\nAdding technical indicators...")
        df_features = add_technical_indicators(df)
        
        if df_features is not None:
             print("Technical indicators added successfully.")
             print(f"Columns: {df_features.columns}")
             print(df_features[['Close', 'MA_10', 'MA_50', 'RSI', 'Volatility']].tail())
        else:
            print("Failed to add technical indicators.")
    else:
        print("Failed to download data.")

if __name__ == "__main__":
    test_data_loader()
