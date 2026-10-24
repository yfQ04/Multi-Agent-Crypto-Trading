# strategies/buy_and_hold.py
"""
Buy & Hold Strategy
- Implement a simple buy-and-hold strategy for Bitcoin
"""

import pandas as pd
from backtest.engine import calculate_metrics

def buy_and_hold(prices: pd.Series) -> dict:
    """
    Implements buy-and-hold strategy and returns full performance metrics.
    """
    returns = prices.pct_change().fillna(0)
    return calculate_metrics(returns)
