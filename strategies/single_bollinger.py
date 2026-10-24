"""
Single Bollinger Strategy
- Long-only mean-reversion strategy based on Bollinger Bands.
"""

import numpy as np
import pandas as pd

from backtest.engine import calculate_metrics


def single_bollinger_position(
    df: pd.DataFrame,
    window: int = 10,
    num_std: float = 2.5,
) -> pd.Series:
    """Build long-only position series from Bollinger bands."""
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain 'Close' column")

    prices = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    middle = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std(ddof=0)
    lower = middle - num_std * std

    pos_values = np.zeros(len(prices), dtype=float)
    in_position = 0.0

    for i in range(len(prices)):
        p = float(prices.iloc[i]) if pd.notna(prices.iloc[i]) else np.nan
        m = float(middle.iloc[i]) if pd.notna(middle.iloc[i]) else np.nan
        l = float(lower.iloc[i]) if pd.notna(lower.iloc[i]) else np.nan

        if np.isnan(p) or np.isnan(m) or np.isnan(l):
            pos_values[i] = in_position
            continue

        # Enter long at/under lower band, exit when mean-reverting back to middle band.
        if in_position <= 0.0 and p <= l:
            in_position = 1.0
        elif in_position > 0.0 and p >= m:
            in_position = 0.0

        pos_values[i] = in_position

    return pd.Series(pos_values, index=prices.index, name="SingleBollingerPosition")


def single_bollinger_strategy(
    df: pd.DataFrame,
    window: int = 10,
    num_std: float = 2.5,
) -> dict:
    """Evaluate Single Bollinger strategy and return full performance metrics."""
    positions = single_bollinger_position(df, window=window, num_std=num_std)
    prices = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    daily_returns = prices.pct_change().fillna(0.0)
    strat_returns = daily_returns * positions.shift(1).fillna(0.0)

    metrics = calculate_metrics(strat_returns)
    metrics["Position series"] = positions
    return metrics
