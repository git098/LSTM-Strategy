import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta

# Load the new data
# The backtesting library expects column names to be capitalized: Open, High, Low, Close
try:
    df = pd.read_csv('data/DABUR.csv', index_col='date', parse_dates=True)
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
except Exception as e:
    print(f"Error loading or processing data: {e}")
    # Create a dummy df to avoid further errors if file is not found
    df = pd.DataFrame()

# Define the Strategy class for the backtest
class MomentumReversionStrategy(Strategy):
    # --- Strategy Parameters ---
    # These will be optimized by the backtester
    rsi_window = 14
    macd_slow = 26
    macd_fast = 12
    macd_sign = 9
    boll_window = 20
    boll_dev = 2

    def init(self):
        # --- Pre-calculate indicators ---
        # This is the correct way to add indicators in backtesting.py
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=self.rsi_window)
        
        # In backtesting.py, ta.macd returns a DataFrame, so we access the columns
        macd_df = self.I(ta.macd, pd.Series(self.data.Close), fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_sign)
        self.macd_line = macd_df.iloc[:, 0] # MACD Line
        self.macd_hist = macd_df.iloc[:, 1] # MACD Histogram
        self.macd_signal = macd_df.iloc[:, 2] # Signal Line

        bbands_df = self.I(ta.bbands, pd.Series(self.data.Close), length=self.boll_window, std=self.boll_dev)
        self.bb_low = bbands_df.iloc[:, 0] # Lower Band
        self.bb_mid = bbands_df.iloc[:, 1] # Middle Band
        self.bb_high = bbands_df.iloc[:, 2] # Upper Band

    def next(self):
        # --- Trading Logic ---
        price = self.data.Close[-1]

        # Buy conditions: price touches lower Bollinger band and RSI is oversold
        if not self.position and price <= self.bb_low[-1] and self.rsi[-1] < 30:
            self.buy()

        # Sell conditions: price touches upper Bollinger band or RSI is overbought
        elif self.position and (price >= self.bb_high[-1] or self.rsi[-1] > 70):
            self.position.close()


if not df.empty:
    # --- Run the Backtest ---
    bt = Backtest(df, MomentumReversionStrategy, cash=100000, commission=.002)

    # --- Optimize the strategy parameters ---
    print("Starting optimization... This may take a while.")
    stats = bt.optimize(
        rsi_window=range(8, 22, 2),
        macd_slow=range(18, 32, 2),
        macd_fast=range(6, 14, 2),
        macd_sign=range(6, 14, 2),
        boll_window=range(10, 31, 2),
        boll_dev=[1.5, 2, 2.5],
        maximize='Sharpe Ratio',
        constraint=lambda p: p.macd_fast < p.macd_slow
    )

    # --- Print and Plot the Results ---
    print("\n--- Optimization Results ---")
    print(stats)

    print("\n--- Best Strategy Trades ---")
    print(stats._trades)

    print("\n--- Plotting Results ---")
    bt.plot()
else:
    print("Could not run backtest because the DataFrame is empty.")
