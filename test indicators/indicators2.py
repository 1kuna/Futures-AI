from datetime import datetime, timedelta
from itertools import combinations
import pandas as pd
import numpy as np
import ta
import yfinance as yf
import matplotlib.pyplot as plt

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
    
    return dataframes

# Calculate indicators
def calculate_indicators(df):
    indicators = {}
    
    # EMAs
    for period in [9, 14, 21]:
        df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        indicators[f'EMA_{period}'] = df[f'EMA_{period}']

    # SMAs
    for period in [21, 50, 100]:
        df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
        indicators[f'SMA_{period}'] = df[f'SMA_{period}']

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'])
    indicators['RSI'] = df['RSI']

    # MACD
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    indicators['MACD'] = df['MACD']

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_mid'] = bollinger.bollinger_mavg()
    df['BB_low'] = bollinger.bollinger_lband()
    indicators['Bollinger_Bands'] = df['BB_mid']

    # ADI
    df['ADI'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
    indicators['ADI'] = df['ADI']

    # OBV
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    indicators['OBV'] = df['OBV']

                            
    # MACD Divergences
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['Buy_Signal_MACD'] = 0
    df['Sell_Signal_MACD'] = 0
    for i in range(1, len(df)):
        if df['Close'].iloc[i] < df['Close'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD'].iloc[i-1]:
            df['Buy_Signal_MACD'].iloc[i] = 1
        elif df['Close'].iloc[i] > df['Close'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD'].iloc[i-1]:
            df['Sell_Signal_MACD'].iloc[i] = 1

    # Stoch RSI Overbought/Oversold Detection
    df['Overbought'] = (df['Stoch_RSI'] > 0.80).astype(int)
    df['Oversold'] = (df['Stoch_RSI'] < 0.20).astype(int)

    # Fibonacci Retracement (Complex and might require a custom function)

    return df, indicators

# Generate signals
def generate_signals(df, indicators):
    signals = {}

    # EMA Crossovers
    for short_period in [9, 14, 21]:
        for long_period in [14, 21]:
            if short_period >= long_period:
                continue
            df[f'signal_EMA{short_period}_EMA{long_period}'] = np.where(
                df[f'EMA_{short_period}'] > df[f'EMA_{long_period}'], 1,
                np.where(df[f'EMA_{short_period}'] < df[f'EMA_{long_period}'], -1, 0)
            )
            signals[f'EMA{short_period}/EMA{long_period}'] = df[f'signal_EMA{short_period}_EMA{long_period}']

    # SMA Crossovers
    for short_period in [21]:
        for long_period in [50, 100]:
            if short_period >= long_period:
                continue
            df[f'signal_SMA{short_period}_SMA{long_period}'] = np.where(
                df[f'SMA_{short_period}'] > df[f'SMA_{long_period}'], 1,
                np.where(df[f'SMA_{short_period}'] < df[f'SMA_{long_period}'], -1, 0)
            )
            signals[f'SMA{short_period}/SMA{long_period}'] = df[f'signal_SMA{short_period}_SMA{long_period}']

    # Stoch RSI
    df['signal_Stoch_RSI'] = np.where(
        df['Stoch_RSI'] > 0.8, -1,
        np.where(df['Stoch_RSI'] < 0.2, 1, 0)
    )
    signals['Stoch_RSI'] = df['signal_Stoch_RSI']

    # RSI
    df['signal_RSI'] = np.where(
        df['RSI'] > 70, -1,
        np.where(df['RSI'] < 30, 1, 0)
    )
    signals['RSI'] = df['signal_RSI']

    # MACD
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['signal_MACD'] = np.where(
        df['MACD'] > df['MACD_signal'], 1,
        np.where(df['MACD'] < df['MACD_signal'], -1, 0)
    )
    signals['MACD'] = df['signal_MACD']

    # Bollinger Bands
    df['signal_Bollinger_Bands'] = np.where(
        df['Close'] < df['BB_low'], 1,
        np.where(df['Close'] > df['BB_high'], -1, 0)
    )
    signals['Bollinger_Bands'] = df['signal_Bollinger_Bands']

    # ADI (Placeholder; more nuanced approach may be needed)
    df['signal_ADI'] = np.where(df['ADI'] > 25, 1, np.where(df['ADI'] < -25, -1, 0))
    signals['ADI'] = df['signal_ADI']

    # OBV (Placeholder; using simple differentiation for now)
    df['OBV_diff'] = df['OBV'].diff()
    df['signal_OBV'] = np.where(df['OBV_diff'] > 0, 1, -1)
    signals['OBV'] = df['signal_OBV']

def generate_combined_signals(df, indicators_list):
    combined_signals = pd.DataFrame(index=df.index)
    
    # Create combinations of indicators (up to 3)
    for r in range(1, 4):
        for combination in combinations(indicators_list, r):
            combined_name = ' / '.join(combination)
            combined_signal = np.ones_like(df['Close'])
            for indicator in combination:
                combined_signal *= df[f'signal_{indicator}']
            combined_signals[combined_name] = combined_signal
    
    return combined_signals

def backtest(df, signals):
    results = pd.DataFrame(index=df.index)

    for name in signals.columns:
        signal = signals[name]
        results[f'return_{name}'] = df['Close'].pct_change() * signal.shift(1)
        results[f'cum_return_{name}'] = (1 + results[f'return_{name}']).cumprod()
    
    return results

# Visualize
def visualize(results, starting_balance=25000):
    plt.figure(figsize=(15,7))

    # Get the final cumulative return for each strategy
    final_returns = {name: cum_return.iloc[-1] for name, cum_return in results.items()}

    # Sort by final return and pick top 5
    top_strategies = sorted(final_returns, key=final_returns.get, reverse=True)[:5]

    for name in top_strategies:
        plt.plot((results[name] - 1) * starting_balance, label=name)

    plt.title('Top 5 Strategy Cumulative P/L in Dollar Amounts over Time')
    plt.xlabel('Date')
    plt.ylabel('P/L in Dollar Amounts')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()