import argparse
import pandas as pd
import numpy as np
from keras.models import load_model
from utils.indicators import add_indicators
from models.lstm_model import create_lstm_model, train_model

def prepare_data_for_lstm(df, window_size):
    """Prepares a dataframe for LSTM training and testing."""
    X, y = [], []
    # Ensure we are targeting the 'close' price, which should be at index 3
    # Columns: open, high, low, close, volume, ...
    close_price_index = 3 
    for i in range(window_size, len(df)):
        X.append(df.iloc[i-window_size:i, :].values)
        y.append(df.iloc[i, close_price_index])
    return np.array(X), np.array(y)

def main():
    # --- Hardcoded Parameters ---
    window_size = 60
    epochs = 10
    model_filename = 'coromandel_trained_model.h5'

    # === PART 1: TRAINING ON COROMANDEL ===
    print("--- Part 1: Training on COROMANDEL data ---")
    
    # 1. Load and Prepare Training Data
    print("Loading and preparing COROMANDEL.csv...")
    df_train = pd.read_csv('data/COROMANDEL.csv', index_col='date', parse_dates=True)
    df_train = add_indicators(df_train)
    df_train = df_train.dropna()

    X_train, y_train = prepare_data_for_lstm(df_train, window_size)

    # 2. Create, Train, and Save the Model
    print(f"Training LSTM model for {epochs} epochs...")
    model = create_lstm_model(X_train.shape[1], X_train.shape[2])
    model = train_model(model, X_train, y_train, epochs=epochs)
    print(f"Training complete. Saving model to {model_filename}...")
    model.save(model_filename)

    # === PART 2: TESTING ON DABUR ===
    print("\n--- Part 2: Testing on DABUR data ---")

    # 3. Load and Prepare Test Data
    print("Loading and preparing DABUR.csv...")
    df_test = pd.read_csv('data/DABUR.csv', index_col='date', parse_dates=True)
    df_test = add_indicators(df_test)
    df_test = df_test.dropna()

    X_test, y_test = prepare_data_for_lstm(df_test, window_size)

    # 4. Load the Trained Model and Make Predictions
    print(f"Loading saved model from {model_filename}...")
    loaded_model = load_model(model_filename)
    print("Generating predictions on DABUR data...")
    predictions = loaded_model.predict(X_test)
    predictions = predictions.flatten() # Make it a 1D array

    # 5. Run Backtesting Simulation on DABUR
    print("Running backtesting simulation on DABUR predictions...")
    initial_cash = 100000
    cash = initial_cash
    position = 0
    trades = []
    equity_curve = [initial_cash]

    for i in range(len(predictions) - 1):
        current_price = y_test[i]
        predicted_price = predictions[i]

        if predicted_price > current_price and cash > current_price:
            if position == 0: # Only buy if we have no position
                shares_to_buy = cash // current_price
                position = shares_to_buy
                cash -= shares_to_buy * current_price
                trades.append({'type': 'buy', 'price': current_price, 'shares': shares_to_buy, 'date': df_test.index[-len(y_test)+i]})

        elif predicted_price < current_price and position > 0:
            cash += position * current_price
            trades.append({'type': 'sell', 'price': current_price, 'shares': position, 'date': df_test.index[-len(y_test)+i]})
            position = 0
        
        current_equity = cash + position * current_price
        equity_curve.append(current_equity)

    # 6. Calculate and Print Performance Metrics
    print("Calculating final performance metrics...")
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

    print("\n--- LSTM Backtest Report (Train: COROMANDEL, Test: DABUR) ---")
    print(f"Initial Equity:       ${initial_cash:,.2f}")
    print(f"Final Equity:         ${final_equity:,.2f}")
    print(f"Total Return:         {total_return_pct:.2f}%")
    print(f"Max Drawdown:         {max_drawdown_pct:.2f}%")
    print(f"Total Trades:         {num_trades}")
    print(f"Win Rate:             {win_rate:.2f}%")
    print("------------------------------------------------------------\n")

if __name__ == '__main__':
    main()
