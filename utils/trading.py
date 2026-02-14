import numpy as np

def get_trading_signal(current_price, predicted_price_next_day):
    """
    Generates a Buy/Sell/Hold signal based on price prediction.
    Returns a dictionary with action, reason, confidence, and risk metrics.
    """
    if current_price == 0:
        return {
            "action": "HOLD",
            "reason": "Current price is zero (Data Error).",
            "confidence": 0.0,
            "stop_loss": 0,
            "take_profit": 0
        }

    threshold = 0.005 # 0.5% threshold to avoid noise
    
    change = (predicted_price_next_day - current_price) / current_price
    abs_change = abs(change)
    
    # Calculate Confidence (0.0 - 1.0)
    # 0.5% change = 50% confidence (Base)
    # > 2.0% change = 90% confidence
    # Logic: map change magnitude to confidence
    confidence = min(max(0.5 + (abs_change - threshold) * 20, 0.5), 0.95)
    
    if change > threshold:
        action = "BUY"
        reason = f"Predicted rise of {change*100:.2f}%"
        take_profit = predicted_price_next_day
        stop_loss = current_price * 0.985 # 1.5% Stop Loss below entry
        
    elif change < -threshold:
        action = "SELL"
        reason = f"Predicted drop of {change*100:.2f}%"
        take_profit = predicted_price_next_day
        stop_loss = current_price * 1.015 # 1.5% Stop Loss above entry (for short)
        
    else:
        action = "HOLD"
        reason = f"Predicted movement {change*100:.2f}% is within noise threshold"
        take_profit = current_price * 1.01
        stop_loss = current_price * 0.99
        confidence = 0.5

    return {
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "predicted_change": change
    }
