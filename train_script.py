import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import create_lstm_model, train_model, get_predictions

# 1. Load and Preprocess the Data
df = pd.read_csv('/Users/himanshuthakur/Desktop/The improvement project/ML_TradingStrategy-main/COROMANDEL.csv')
data = df['close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 2. Create Training and Test Sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

n_steps = 60

def create_dataset(dataset, n_steps):
    X, y = [], []
    for i in range(n_steps, len(dataset)):
        X.append(dataset[i-n_steps:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, n_steps)
X_test, y_test = create_dataset(test_data, n_steps)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3. Build and Train the LSTM Model
window_size = n_steps
n_features = 1
epochs = 25

lstm_model = create_lstm_model(window_size, n_features)

print(f"Training model for {epochs} epochs...")
trained_model = train_model(lstm_model, X_train, y_train, epochs=epochs, batch_size=32)
print("Model training complete.")

# 4. Performance Evaluation
predictions = get_predictions(trained_model, X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 5. Simulated Trading
trades = []
balance = 100000  # Initial balance
position = None

for i in range(len(predictions) - 1):
    if predictions[i+1] > y_test_actual[i] and position != 'long':
        trades.append(('buy', y_test_actual[i][0]))
        position = 'long'
    elif predictions[i+1] < y_test_actual[i] and position == 'long':
        trades.append(('sell', y_test_actual[i][0]))
        position = None

profit = 0
for i in range(0, len(trades) - 1, 2):
    profit += trades[i+1][1] - trades[i][1]

profit_percentage = (profit / balance) * 100

print("\n--- Performance Evaluation ---")
print(f"Total Trades: {len(trades)}")
print(f"Profit: ${profit:.2f}")
print(f"Profit Percentage: {profit_percentage:.2f}%")

# 6. (Optional) Save the trained model
trained_model.save('coromandel_lstm_model.h5')
print("\nTrained model saved to coromandel_lstm_model.h5")