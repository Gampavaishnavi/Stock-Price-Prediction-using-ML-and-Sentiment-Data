import pandas_datareader.data as web
import datetime

start = datetime.datetime(2023, 1, 1)
end = datetime.datetime(2023, 1, 10)

try:
    print("Testing pandas_datareader with Stooq...")
    df = web.DataReader('AAPL.US', 'stooq', start, end)
    print("Stooq success!")
    print(df.head())
except Exception as e:
    print(f"Stooq failed: {e}")

try:
    print("\nTesting pandas_datareader with Yahoo...")
    # Yahoo often fails without yfinance override, but let's try
    df = web.DataReader('AAPL', 'yahoo', start, end)
    print("Yahoo success!")
    print(df.head())
except Exception as e:
    print(f"Yahoo failed: {e}")
