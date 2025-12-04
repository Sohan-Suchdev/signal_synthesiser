import pandas as pd
import numpy as np

def get_meta_labels(df, window=5, barrier=0.015):
    """
    Triple Barrier Method
    
    Target:
    1 (Profitable): If price goes up by 'barrier' % within 'window' days.
    0 (Unprofitable): If it does not.
    """
    df = df.copy()
    
    # Look forward using a rolling max
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    future_max = df['High'].rolling(window=indexer).max()
    
    target_price = df['Close'] * (1 + barrier)
    
    # Create the binary label
    df['Target_Label'] = np.where(future_max > target_price, 1, 0)
    
    df = df.iloc[:-window]
    
    return df