
from utils.data_loader import load_data
from utils.indicators import add_indicators
from models.lstm_model import create_lstm_model, train_model, get_predictions
from utils.signals import generate_signals
from backtests.engine import run_backtest
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Load data
    df = load_data(['AAPL', 'GOOG'], '2019-01-01', '2024-07-01')

    # Add indicators
    df = add_indicators(df)
    df = df.dropna()

    # Prepare data for LSTM
    X = []
    y = []
    window = 60
    for i in range(window, len(df)):
        X.append(df.iloc[i-window:i, :].values)
        y.append(df.iloc[i, 1])

    X, y = pd.DataFrame(X), pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    model = create_lstm_model(X_train.shape[1], X_train.shape[2])
    model = train_model(model, X_train, y_train)

    # Get predictions and generate signals
    predictions = get_predictions(model, X_test)
    signals = generate_signals(df.iloc[-len(X_test):], predictions, threshold=0.01)

    # Run backtest
    backtest_results = run_backtest(df.iloc[-len(X_test):], signals, stop_loss=0.05, take_profit=0.1)

    print("Backtest Results:")
    for key, value in backtest_results.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()
