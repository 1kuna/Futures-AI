from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

def gather_timeframe_data(timeframes=None):
    """
    Fetches historical market data for the ES futures from Yahoo Finance for multiple timeframes.
    For timeframes > 1m, it fetches data up to 59 days back.
    For the 1m timeframe, it fetches data up to 6 days back.
    """
    timeframes = timeframes or ["1m", "5m", "15m", "30m", "1h"]
    
    dataframes = {}
    
    end_date = datetime.today()
    for tf in timeframes:
        if tf == "1m":
            start_date = end_date - timedelta(days=6)
        else:
            start_date = end_date - timedelta(days=59)
        
        dataframes[tf] = yf.download("ES=F", start=start_date, end=end_date, interval=tf)
        pd.DataFrame(dataframes[tf]).to_csv(f"K:\\Git\\KUNA\\test indicators\\es_{tf}.csv")

gather_timeframe_data()