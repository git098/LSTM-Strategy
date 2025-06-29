
import pytest
import numpy as np
from models.lstm_model import create_lstm_model, train_model, get_predictions

def test_lstm_model():
    # Create dummy data
    X_train = np.random.rand(100, 60, 1)
    y_train = np.random.rand(100, 1)
    X_test = np.random.rand(20, 60, 1)

    # Create and train model
    model = create_lstm_model(60, 1)
    model = train_model(model, X_train, y_train, epochs=1)

    # Get predictions
    predictions = get_predictions(model, X_test)

    assert predictions.shape == (20, 1)
