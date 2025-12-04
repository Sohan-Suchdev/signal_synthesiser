import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker, start_date, end_date, data_dir="data/raw"):
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    
    # Cache Check
    if os.path.exists(file_path):
        print(f"Loading {ticker} from cache...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {ticker} from yfinance...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # yfinance sometimes returns MultiIndex columns, flatten them if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.to_csv(file_path)
        print(f"Saved {ticker} to {file_path}")

    # Basic Cleaning
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
        
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]