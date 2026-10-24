# strategies/sma_strategy.py
"""
SMA Strategy
- Implements a simple moving average trading strategy
"""

import pandas as pd
import numpy as np
from backtest.engine import calculate_metrics

def sma_strategy(df: pd.DataFrame, short_window: int = 7, long_window: int = 20) -> dict:
    """
    Implements a simple moving average crossover strategy.
    """
    prices = df["Close"]
    sma_short = prices.rolling(window=short_window).mean()
    sma_long = prices.rolling(window=long_window).mean()

    # Create signal: 1 if short MA > long MA else 0
    signals = np.where(sma_short > sma_long, 1, 0)
    positions = pd.Series(signals, index=prices.index)
    
    # Calculate daily returns
    daily_returns = prices.pct_change().fillna(0)
    
    # Strategy returns (shift position by 1 day to avoid look-ahead bias)
    strat_returns = daily_returns * positions.shift(1).fillna(0)
    
    return calculate_metrics(strat_returns)
