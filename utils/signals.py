
import pandas as pd

def generate_signals(df, predictions, threshold=0.01):
    """
    Generates trading signals based on the model's predictions.
    """
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['predicted_return'] = predictions

    # Buy signal: predicted return is above the threshold
    signals.loc[signals['predicted_return'] > threshold, 'signal'] = 1

    # Sell signal: predicted return is below the negative threshold
    signals.loc[signals['predicted_return'] < -threshold, 'signal'] = -1

    # Add explanation
    signals['explanation'] = 'Hold'
    signals.loc[signals['signal'] == 1, 'explanation'] = 'Buy signal: predicted return is positive'
    signals.loc[signals['signal'] == -1, 'explanation'] = 'Sell signal: predicted return is negative'

    return signals
