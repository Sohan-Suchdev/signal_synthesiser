import pandas as pd
import numpy as np

def calculate_features(df):
    """
    Technical indicators:
    - RSI (Mean Reversion)
    - MACD (Momentum)
    - Bollinger Bands (Volatility Breakout)
    - Trend (SMA Distance)
    - Volatility (Standard Deviation)
    """
    df = df.copy()
    
    # 1. Daily Returns 
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Volatility (20-day Rolling Std Dev of returns)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # 3. RSI (Relative Strength Index) - 14 Day
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Trend: Distance from Moving Average (SMA 50)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Trend_Score'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # 5. MACD (Moving Average Convergence Divergence) - Momentum
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    
    # 6. Bollinger Bands - Volatility Breakout
    # 2 Standard Deviations above/below the 20-day average.
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    
    # DROP NaNs 
    df.dropna(inplace=True)
    
    return df