import pandas as pd

class TradingAgent:
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.holdings = 0
        self.portfolio_value = initial_balance
        self.trade_history = []
        self.history = [] # Tracks daily portfolio value

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        self.history = []

    def execute_trade(self, date, signal, price, reason="", quantity=None, volatility=0.0, sentiment=0.0):
        """
        Executes a trade based on the signal ('BUY', 'SELL', 'HOLD').
        Enhanced with Volatility, Sentiment, and Risk Management logic.
        """
        # --- Advanced Decision Logic ---
        
        # 1. Volatility Filter: High volatility -> High Risk -> NO TRADE
        VOLATILITY_THRESHOLD = 0.025 # 2.5% daily move
        if volatility > VOLATILITY_THRESHOLD:
            # We skip the trade but log it
            self.trade_history.append({
                "Date": date,
                "Action": "SKIP",
                "Price": price,
                "Shares": 0,
                "Value": 0,
                "Reason": f"High Volatility ({volatility:.3f})"
            })
            return # Exit early

        # 2. Risk Management: Position Sizing (Max 10% of portfolio per trade if auto-trade)
        MAX_RISK_PER_TRADE = 0.10
        
        # 3. Sentiment Weighting (Boost confidence)
        if signal == "BUY" and self.balance > 0:
            # Determine shares to buy
            if quantity is not None:
                shares_to_buy = int(quantity)
            else:
                # Auto-Trade: Buy max 10% of portfolio value or max affordable
                target_investment = self.portfolio_value * MAX_RISK_PER_TRADE
                shares_to_buy = int(min(self.balance, target_investment) // price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                
                # Check affordability
                if cost <= self.balance:
                    self.balance -= cost
                    self.holdings += shares_to_buy
                    self.trade_history.append({
                        "Date": date,
                        "Action": "BUY",
                        "Price": price,
                        "Shares": shares_to_buy,
                        "Value": cost,
                        "Reason": reason + f" | Vol:{volatility:.3f} Sent:{sentiment:.1f}"
                    })
                else:
                    print("Insufficient funds for requested quantity.")
        
        elif signal == "SELL" and self.holdings > 0:
            # Determine shares to sell
            if quantity is not None:
                shares_to_sell = int(quantity)
            else:
                shares_to_sell = self.holdings 
                
            if shares_to_sell > 0:
                # Check ownership
                if shares_to_sell <= self.holdings:
                    revenue = shares_to_sell * price
                    self.balance += revenue
                    self.trade_history.append({
                        "Date": date,
                        "Action": "SELL",
                        "Price": price,
                        "Shares": shares_to_sell,
                        "Value": revenue,
                        "Reason": reason + f" | Vol:{volatility:.3f} Sent:{sentiment:.1f}"
                    })
                    self.holdings -= shares_to_sell
                else:
                    print("Insufficient holdings for requested quantity.")
            
        # Update portfolio value
        current_val = self.balance + (self.holdings * price)
        self.portfolio_value = current_val
        
        # Risk Metric: Max Drawdown Update
        if not hasattr(self, 'peak_value'):
            self.peak_value = self.initial_balance
            
        if current_val > self.peak_value:
            self.peak_value = current_val
            
        drawdown = (self.peak_value - current_val) / self.peak_value if self.peak_value > 0 else 0
        
        # Log daily state
        self.history.append({
            "Date": date,
            "Balance": self.balance,
            "Holdings": self.holdings,
            "Stock_Price": price,
            "Portfolio_Value": self.portfolio_value,
            "Drawdown": drawdown
        })
        
    def get_portfolio_history(self):
        return pd.DataFrame(self.history)
        
    def get_trade_history(self):
        return pd.DataFrame(self.trade_history)
