# agents/technical_agent.py
"""agents/technical_agent.py

Agent 2: Crypto Historical Data Retrieval + Technical Indicators
- Fetch OHLCV candles from Coinbase Exchange public API
- Calculate RSI, MACD, Bollinger Bands, ATR, SMA, EMA, Volatility, etc.
- Optionally use SiliconFlow LLM to interpret technical signals
"""

import pandas as pd
import numpy as np
import ta
import requests
from pathlib import Path
from agents.llm_utils import ask_llm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _symbol_to_asset_label(symbol: str) -> str:
    base = (symbol or "").split("-")[0].strip()
    return base if base else symbol


def _to_coinbase_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return "BTC-USD"
    return s if "-" in s else f"{s}-USD"


def _period_to_days(period: str) -> int | None:
    if not period:
        return None
    period = period.strip().lower()
    predefined = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "max": None,
    }
    if period in predefined:
        return predefined[period]
    if period.endswith("d") and period[:-1].isdigit():
        return int(period[:-1])
    return None


def _apply_period_filter(df: pd.DataFrame, period: str) -> pd.DataFrame:
    days = _period_to_days(period)
    if days is None or df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=days)
    filtered = df[df.index >= cutoff]
    return filtered if not filtered.empty else df


def _apply_date_range_filter(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if df.empty or (not start_date and not end_date):
        return df

    filtered = df
    if start_date:
        start_ts = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_ts):
            filtered = filtered[filtered.index >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_ts):
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered = filtered[filtered.index <= end_ts]

    return filtered


def _apply_requested_time_filter(
    df: pd.DataFrame,
    period: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if start_date or end_date:
        return _apply_date_range_filter(df, start_date=start_date, end_date=end_date)
    return _apply_period_filter(df, period)


def _normalize_ohlcv_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=OHLCV_COLUMNS)


def _load_local_data(
    symbol: str = "BTC",
    period: str = "3mo",
    start_date: str | None = None,
    end_date: str | None = None,
    preferred_csv_path: str | None = None,
) -> pd.DataFrame | None:
    base = (symbol or "").split("-")[0].strip().upper()
    if base not in {"BTC", "ETH"}:
        return None

    candidates = []
    if preferred_csv_path:
        preferred_path = Path(preferred_csv_path)
        if not preferred_path.is_absolute():
            preferred_path = PROJECT_ROOT / preferred_path
        candidates.append(preferred_path)

    candidates.extend(
        [
            PROJECT_ROOT / f"{base}.csv",
            PROJECT_ROOT / "data" / f"{base}.csv",
            PROJECT_ROOT / "datasets" / f"{base}.csv",
        ]
    )

    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            raw_df = pd.read_csv(csv_path)
            normalized = {c.lower().strip(): c for c in raw_df.columns}
            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(set(normalized.keys())):
                print(f"[Agent2] Local CSV {csv_path.name} missing required columns, skipping...")
                continue

            df = raw_df.rename(
                columns={
                    normalized["date"]: "Open_time",
                    normalized["open"]: "Open",
                    normalized["high"]: "High",
                    normalized["low"]: "Low",
                    normalized["close"]: "Close",
                    normalized["volume"]: "Volume",
                }
            )

            df["Open_time"] = pd.to_datetime(df["Open_time"], errors="coerce")
            df.dropna(subset=["Open_time"], inplace=True)
            df.set_index("Open_time", inplace=True)
            df.sort_index(ascending=True, inplace=True)

            df = _normalize_ohlcv_numeric(df)
            df = _apply_requested_time_filter(df, period=period, start_date=start_date, end_date=end_date)
            if df.empty:
                print(f"[Agent2] Local CSV {csv_path.name} has no rows in requested date range, skipping...")
                continue
            print(f"[Agent2] Loaded {len(df)} rows for {base} from local CSV: {csv_path}")
            return df
        except Exception as e:
            print(f"[Agent2] Failed reading local CSV {csv_path}: {e}")

    return None


def fetch_coinbase_data(
    symbol: str = "BTC",
    period: str = "3mo",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    preferred_csv_path: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV candles using real data only (local CSV first, then Coinbase API)."""
    local_df = _load_local_data(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        preferred_csv_path=preferred_csv_path,
    )
    if local_df is not None:
        return local_df

    coinbase_symbol = _to_coinbase_symbol(symbol)
    print(f"[Agent2] Downloading {coinbase_symbol} data from Coinbase (interval={interval})...")

    # Coinbase API uses seconds for interval (86400 = 1 day)
    # NOTE: period is currently unused because this endpoint returns a fixed max window.
    url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/candles?granularity=86400"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("Coinbase returned empty data.")

        # Coinbase candle format:
        # [ time, low, high, open, close, volume ]
        # NOTE: data is returned from NEWEST to OLDEST
        df = pd.DataFrame(data, columns=["Open_time", "Low", "High", "Open", "Close", "Volume"])

        df["Open_time"] = pd.to_datetime(df["Open_time"], unit="s")
        df.set_index("Open_time", inplace=True)

        # Sort ascending (oldest to newest) for technical indicators
        df.sort_index(ascending=True, inplace=True)

        df = _normalize_ohlcv_numeric(df)
        df = _apply_requested_time_filter(df, period=period, start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError("No Coinbase rows available in requested date range.")

        print(f"[Agent2] Retrieved {len(df)} records of candlestick data from Coinbase ✅")
        return df

    except Exception as e:
        raise RuntimeError(
            f"[Agent2] Real data unavailable for {symbol}. "
            f"Please provide local CSV ({_symbol_to_asset_label(symbol)}.csv) or ensure Coinbase API is reachable. "
            f"Original error: {e}"
        )


def fetch_bitcoin_data(
    period: str = "3mo",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    preferred_csv_path: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return fetch_coinbase_data(
        symbol="BTC",
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        preferred_csv_path=preferred_csv_path,
    )

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical analysis indicators"""
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    row_count = len(df)
    sma_7_w = max(2, min(7, row_count))
    sma_20_w = max(2, min(20, row_count))
    sma_50_w = max(2, min(50, row_count))
    ema_12_w = max(2, min(12, row_count))
    ema_26_w = max(2, min(26, row_count))
    rsi_14_w = max(2, min(14, row_count))
    bb_20_w = max(2, min(20, row_count))
    atr_14_w = max(2, min(14, row_count))
    vol_20_w = max(2, min(20, row_count))
    vol_sma_20_w = max(2, min(20, row_count))
    momentum_10_p = max(1, min(10, row_count - 1)) if row_count > 1 else 1

    df["SMA_7"] = ta.trend.sma_indicator(close, window=sma_7_w)
    df["SMA_20"] = ta.trend.sma_indicator(close, window=sma_20_w)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=sma_50_w)
    df["EMA_12"] = ta.trend.ema_indicator(close, window=ema_12_w)
    df["EMA_26"] = ta.trend.ema_indicator(close, window=ema_26_w)

    macd_slow = max(4, min(26, row_count))
    macd_fast = max(2, min(12, macd_slow - 1))
    macd_signal = max(2, min(9, macd_fast))
    macd = ta.trend.MACD(close, window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    df["RSI_14"] = ta.momentum.rsi(close, window=rsi_14_w)

    bb = ta.volatility.BollingerBands(close, window=bb_20_w, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()

    df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=atr_14_w)
    df["Returns"] = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(window=vol_20_w).std() * np.sqrt(252) * 100  # Annualized
    df["Volume_SMA_20"] = ta.trend.sma_indicator(volume, window=vol_sma_20_w)
    df["Volume_ratio"] = volume / df["Volume_SMA_20"]
    df["Momentum_10"] = close.pct_change(periods=momentum_10_p) * 100

    df = df[df["Close"].notna()].copy()
    return df

def llm_technical_analysis(df: pd.DataFrame, symbol: str = "BTC") -> str:
    """Interpret latest technical indicators using LLM (optional)."""
    latest = df.iloc[-1]
    close_val = float(latest["Close"])
    asset_label = _symbol_to_asset_label(symbol)

    prompt = f"""
You are a professional quantitative trader, please analyze the latest {asset_label} technical data based on the following indicators:

📊 Latest Data ({df.index[-1]}):
- Current Price: ${close_val:.2f}
- SMA_7: ${float(latest['SMA_7']):.2f} | SMA_20: ${float(latest['SMA_20']):.2f} | SMA_50: ${float(latest['SMA_50']):.2f}
- RSI(14): {float(latest['RSI_14']):.1f} (>70 Overbought, <30 Oversold)
- MACD: {float(latest['MACD']):.2f} | Signal: {float(latest['MACD_signal']):.2f} | Hist: {float(latest['MACD_hist']):.2f}
- Bollinger Bands: Upper ${float(latest['BB_upper']):.2f} / Middle ${float(latest['BB_middle']):.2f} / Lower ${float(latest['BB_lower']):.2f}

Please provide your analysis based on:
1. 📈 Trend Analysis (Short-term/Mid-term)
2. 🔄 Momentum Signals (RSI + MACD Composite)
3. 💹 Volatility Environment (High/Low Volatility)
4. 📊 Key Support/Resistance Levels
5. 🎯 Technical Signal Direction: 【Bullish / Neutral / Bearish】

Please respond in English in a professional manner.
"""
    return ask_llm(prompt)

def run_technical_agent(
    period: str = "3mo",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    preferred_csv_path: str | None = None,
) -> dict:
    """Backward-compatible wrapper."""
    return run_technical_agent_for_symbol(
        symbol="BTC",
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        preferred_csv_path=preferred_csv_path,
    )


def run_technical_agent_for_symbol(
    symbol: str = "BTC",
    period: str = "3mo",
    interval: str = "1d",
    use_llm: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    preferred_csv_path: str | None = None,
) -> dict:
    """Run Agent 2 for a given symbol and return technical analysis results."""
    print("\n" + "="*60)
    print("🤖 Agent 2: Technical Indicator Analysis Starting...")
    print("="*60)

    df = fetch_coinbase_data(
        symbol=symbol,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        preferred_csv_path=preferred_csv_path,
    )
    df = calculate_indicators(df)

    latest = df.iloc[-1]
    close_val = float(latest["Close"])

    if use_llm:
        gpt_analysis = llm_technical_analysis(df, symbol=symbol)
        print("\n[Agent2] Technical analysis report from LLM:")
        print(gpt_analysis)
    else:
        gpt_analysis = "SKIPPED (use_llm=False)"

    return {
        "symbol": symbol,
        "df": df,
        "current_price": close_val,
        "rsi": float(latest["RSI_14"]),
        "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACD_signal"]),
        "volatility": float(latest["Volatility"]),
        "sma_7": float(latest["SMA_7"]),
        "sma_20": float(latest["SMA_20"]),
        "sma_50": float(latest["SMA_50"]),
        "gpt_analysis": gpt_analysis,
    }

if __name__ == "__main__":
    result = run_technical_agent(period="3mo")
    print(f"\nCurrent Price: ${result['current_price']:,.2f}")
    print(f"RSI: {result['rsi']:.1f}")
    print(f"Volatility: {result['volatility']:.1f}%")
