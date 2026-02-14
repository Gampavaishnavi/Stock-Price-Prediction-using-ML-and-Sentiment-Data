try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found. LSTM model will not be available.")

def create_lstm_model(input_shape):
    """
    Creates and compiles the LSTM model.
    Returns None if TensorFlow is not available.
    """
    if not TF_AVAILABLE:
        print("TensorFlow is not installed. Returning None.")
        return None

    model = Sequential()
    # First LSTM layer with Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
