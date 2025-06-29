import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data
from utils.indicators import add_indicators
from models.lstm_model import create_lstm_model, train_model, get_predictions
from utils.signals import generate_signals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='COROMANDEL', help='Stock ticker to train on')
    parser.add_argument('--window', type=int, default=60, help='Window size for LSTM')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--threshold', type=float, default=0.0, help='Signal threshold for buy/sell')
    args = parser.parse_args()

    # --- 1. Load and Prepare Data ---
    print(f"Loading data for {args.ticker}...")
    # df = load_data([args.ticker], '2019-01-01', '2024-07-01')
    df = pd.read_csv('data/COROMANDEL.csv', index_col='date', parse_dates=True)
    
    print("Adding technical indicators...")
    df = add_indicators(df)
    df = df.dropna()

    # --- 2. Prepare Data for LSTM ---
    print("Preparing data for LSTM model...")
    X = []
    y = []
    for i in range(args.window, len(df)):
        X.append(df.iloc[i-args.window:i, :].values)
        y.append(df.iloc[i, 3]) # Target the 'close' price (index 3)

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # --- 3. Create and Train Model ---
    print("Creating and training the LSTM model...")
    model = create_lstm_model(X_train.shape[1], X_train.shape[2])
    model = train_model(model, X_train, y_train, epochs=args.epochs)
    print("Training complete. Saving model...")
    model.save('coromandel_lstm_model.h5')

    # --- 4. Get Predictions ---
    print("Generating predictions on the test set...")
    predictions = get_predictions(model, X_test)
    predictions = predictions.flatten() # Make it a 1D array

    # --- 5. Backtesting Simulation ---
    print("Running backtesting simulation...")
    initial_cash = 100000
    cash = initial_cash
    position = 0 # Number of shares held
    trades = []
    equity_curve = [initial_cash]

    for i in range(len(predictions) - 1):
        current_price = y_test[i]
        predicted_price = predictions[i]

        # Buy Signal
        if predicted_price > current_price and position == 0:
            shares_to_buy = cash // current_price
            position = shares_to_buy
            cash -= shares_to_buy * current_price
            trades.append({'type': 'buy', 'price': current_price, 'shares': shares_to_buy, 'date': df.index[-len(y_test)+i]})

        # Sell Signal
        elif predicted_price < current_price and position > 0:
            cash += position * current_price
            trades.append({'type': 'sell', 'price': current_price, 'shares': position, 'date': df.index[-len(y_test)+i]})
            position = 0
        
        # Update equity curve
        current_equity = cash + position * current_price
        equity_curve.append(current_equity)

    # --- 6. Performance Metrics Calculation ---
    print("Calculating performance metrics...")
    final_equity = equity_curve[-1]
    total_return_pct = ((final_equity - initial_cash) / initial_cash) * 100
    
    pnls = []
    for i in range(0, len(trades) - 1, 2):
        if trades[i]['type'] == 'buy' and trades[i+1]['type'] == 'sell':
            pnl = (trades[i+1]['price'] - trades[i]['price']) * trades[i]['shares']
            pnls.append(pnl)

    num_trades = len(pnls)
    win_rate = (len([p for p in pnls if p > 0]) / num_trades * 100) if num_trades > 0 else 0
    
    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown_pct = drawdown.min() * 100

    # --- 7. Print Performance Report ---
    print("\n--- LSTM Model Backtest Report ---")
    print(f"Initial Equity:       ${initial_cash:,.2f}")
    print(f"Final Equity:         ${final_equity:,.2f}")
    print(f"Total Return:         {total_return_pct:.2f}%")
    print(f"Max Drawdown:         {max_drawdown_pct:.2f}%")
    print(f"Total Trades:         {num_trades}")
    print(f"Win Rate:             {win_rate:.2f}%")
    print("------------------------------------\n")

if __name__ == '__main__':
    main()