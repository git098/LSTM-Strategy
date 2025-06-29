
import pytest
from utils.data_loader import load_data

def test_load_data():
    df = load_data(['AAPL'], '2023-01-01', '2023-01-31')
    assert not df.empty
    assert 'adj close' in df.columns
