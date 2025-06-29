import pandas as pd
import numpy as np
import pandas_ta as ta

def add_indicators(df):
    """
    Adds technical indicators to the dataframe.
    """
    # Use 'close' column from the CSV and remove groupby
    df['rsi'] = ta.rsi(close=df['close'], length=20)
    
    # Calculate bbands once and assign to columns
    bbands = ta.bbands(close=np.log1p(df['close']), length=20)
    if bbands is not None and not bbands.empty:
        df['bb_low'] = bbands.iloc[:,0]
        df['bb_mid'] = bbands.iloc[:,1]
        df['bb_high'] = bbands.iloc[:,2]
    else:
        df['bb_low'] = np.nan
        df['bb_mid'] = np.nan
        df['bb_high'] = np.nan

    def compute_atr(stock_data):
        atr = ta.atr(high=stock_data['high'],
                            low=stock_data['low'],
                            close=stock_data['close'],
                            length=14)
        return atr.sub(atr.mean()).div(atr.std())

    # Apply directly to the dataframe
    df['atr'] = compute_atr(df)
    
    def compute_macd(close):
        macd = ta.macd(close=close, length=20)
        if macd is not None and not macd.empty:
            return macd.iloc[:,0].sub(macd.iloc[:,0].mean()).div(macd.iloc[:,0].std())
        return np.nan

    # Apply directly to the 'close' column
    df['macd'] = compute_macd(df['close'])
    
    # Add Stochastic Oscillator
    def compute_stoch(stock_data):
        stoch = ta.stoch(high=stock_data['high'],
                         low=stock_data['low'],
                         close=stock_data['close'],
                         k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            return stoch.iloc[:,0]
        return np.nan

    # Apply directly to the dataframe
    df['stoch'] = compute_stoch(df)
    
    # Use 'close' column
    df['rupee_volume'] = (df['close'] * df['volume']) / 1e6
    
    return df