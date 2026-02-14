from alpaca_trade_api.rest import REST, TimeFrame
import os

class AlpacaBroker:
    def __init__(self, key_id, secret_key, base_url='https://paper-api.alpaca.markets'):
        try:
            self.api = REST(key_id, secret_key, base_url=base_url)
            self.account = self.api.get_account()
            if self.account.status != 'ACTIVE':
                print(f"Alpaca Account Status: {self.account.status}")
            else:
                print("Alpaca Connected Successfully.")
        except Exception as e:
            print(f"Alpaca Connection Failed: {e}")
            self.api = None
            raise e

    def get_balance(self):
        """Returns the current simulated cash balance."""
        if not self.api: return 0.0
        try:
            account = self.api.get_account()
            return float(account.cash)
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def get_equity(self):
        """Returns the current total equity."""
        if not self.api: return 0.0
        try:
            account = self.api.get_account()
            return float(account.equity)
        except Exception as e:
            print(f"Error fetching equity: {e}")
            return 0.0

    def get_positions(self):
        """Returns a list of open positions."""
        if not self.api: return []
        try:
            positions = self.api.list_positions()
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def submit_order(self, symbol, qty, side, type='market', time_in_force='gtc'):
        """
        Submits an order to Alpaca.
        side: 'buy' or 'sell'
        """
        if not self.api: return None
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force
            )
            print(f"Order Submitted: {side} {qty} {symbol}")
            return order
        except Exception as e:
            print(f"Order Failed: {e}")
            return None

    def get_status(self):
        return True if self.api else False
