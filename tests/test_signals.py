
import pytest
import pandas as pd
import numpy as np
from utils.signals import generate_signals

def test_generate_signals():
    # Create a dummy dataframe and predictions
    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=5))
    df = pd.DataFrame(index=pd.MultiIndex.from_product([dates, ['AAPL']], names=['date', 'ticker']))
    predictions = np.array([0.1, -0.2, 0.3, -0.4, 0.5])

    signals = generate_signals(df, predictions)

    assert not signals.empty
    assert 'signal' in signals.columns
    assert 'explanation' in signals.columns
    assert signals['signal'].iloc[0] == 1
    assert signals['signal'].iloc[1] == -1
