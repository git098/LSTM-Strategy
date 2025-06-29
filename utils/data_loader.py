
import pandas as pd
import yfinance as yf

def load_data(tickers, start_date, end_date):
    """
    Downloads and stacks data for a list of tickers.
    """
    df = yf.download(tickers=tickers, start=start_date, end=end_date).stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    return df
