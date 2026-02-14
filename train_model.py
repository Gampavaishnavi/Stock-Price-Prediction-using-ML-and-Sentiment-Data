import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pickle
import os
import joblib

from utils.data_loader import download_data, add_technical_indicators
from sentiment.analyzer import SentimentAnalyzer

# Constants
TICKER = 'AAPL'
START_DATE = '2018-01-01'
END_DATE = '2024-01-01'
SEQUENCE_LENGTH = 60
MODEL_PATH = "models/mlp_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_and_preprocess_data(ticker, start, end):
    print(f"Loading data for {ticker}...")
    df = download_data(ticker, start, end)
    
    if df is None or df.empty:
        raise ValueError("Could not fetch data.")
        
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    print("Adding sentiment data (Simulated)...")
    analyzer = SentimentAnalyzer()
    
    # Check bounds
    data_start = df['Date'].min()
    data_end = df['Date'].max()
    sentiment_df = analyzer.simulate_daily_sentiment(data_start, data_end)
    
    # Merge sentiment on Date
    df['Date'] = pd.to_datetime(df['Date'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    
    df = pd.merge(df, sentiment_df, on='Date', how='left')
    df['Sentiment'].fillna(0, inplace=True)
    
    return df

def prepare_training_data(df, features, sequence_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        # Flatten the sequence window: (60, num_features) -> (60 * num_features)
        window = scaled_data[i-sequence_length:i]
        X.append(window.flatten())
        y.append(scaled_data[i][0]) # Predict Close price (index 0)
        
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train():
    if not os.path.exists("models"):
        os.makedirs("models")

    # 1. Load Data
    try:
        df = load_and_preprocess_data(TICKER, START_DATE, END_DATE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Use Features
    features = ['Close', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'Sentiment']
    # Check if RSI is present
    features = [f for f in features if f in df.columns]
    
    print(f"Training on features: {features}")
    
    # 2. Prepare Data
    X, y, scaler = prepare_training_data(df, features, SEQUENCE_LENGTH)
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("Not enough data to train. Exiting.")
        return

    # 3. Build & Train MLP (ANN) Model
    print("Building MLP Regressor (ANN)...")
    # Hidden layers: 64 neurons, 32 neurons. ReLU activation. 
    model = MLPRegressor(hidden_layer_sizes=(64, 32), 
                         activation='relu', 
                         solver='adam', 
                         max_iter=500, 
                         random_state=42, 
                         verbose=True)
    
    print("Starting training...")
    model.fit(X, y)
    
    print(f"Training score: {model.score(X, y)}")
    
    # 4. Save Model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    # 5. Save Scaler
    print(f"Saving scaler to {SCALER_PATH}...")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    print("Training process complete!")

if __name__ == "__main__":
    train()
