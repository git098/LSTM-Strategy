import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_lstm_model(window_size, n_features):
    """
    Creates the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Trains the LSTM model.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def get_predictions(model, X_test):
    """
    Makes predictions with the LSTM model.
    """
    return model.predict(X_test)