
import pandas as pd
import numpy as np

def run_backtest(df, signals, stop_loss=0.05, take_profit=0.1):
    """
    Runs a backtest of the trading strategy.
    """
    initial_capital = float(100000.0)

    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['stock'] = signals['signal']

    # Apply stop-loss and take-profit
    for i in range(1, len(portfolio)):
        if positions.iloc[i-1]['stock'] == 1 and (df.iloc[i]['adj close'] <= df.iloc[i-1]['adj close'] * (1 - stop_loss) or df.iloc[i]['adj close'] >= df.iloc[i-1]['adj close'] * (1 + take_profit)):
            positions.iloc[i]['stock'] = 0
        elif positions.iloc[i-1]['stock'] == -1 and (df.iloc[i]['adj close'] >= df.iloc[i-1]['adj close'] * (1 + stop_loss) or df.iloc[i]['adj close'] <= df.iloc[i-1]['adj close'] * (1 - take_profit)):
            positions.iloc[i]['stock'] = 0

    portfolio = positions.multiply(df['adj close'], axis=0)

    pos_diff = positions.diff()

    portfolio['holdings'] = (positions.multiply(df['adj close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(df['adj close'], axis=0)).sum(axis=1).cumsum()

    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    # Calculate Sharpe Ratio
    sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())

    # Calculate max drawdown
    window = 252
    rolling_max = portfolio['total'].rolling(window, min_periods=1).max()
    daily_drawdown = portfolio['total']/rolling_max - 1.0
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

    return {
        'portfolio_value': portfolio['total'],
        'pnl': portfolio['total'][-1] - initial_capital,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_daily_drawdown.min()
    }
