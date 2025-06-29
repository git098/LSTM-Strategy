
import pytest
import pandas as pd
from utils.data_loader import load_data
from utils.indicators import add_indicators

def test_add_indicators():
    df = load_data(['AAPL'], '2023-01-01', '2023-01-31')
    df = add_indicators(df)
    assert 'rsi' in df.columns
    assert 'bb_low' in df.columns
    assert 'stoch' in df.columns
