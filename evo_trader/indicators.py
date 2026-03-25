"""
Technical indicators computed on price data (pandas Series/DataFrame).
All functions take a pandas DataFrame with at least a 'Close' column.
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Returns (middle, upper, lower) Bollinger Bands."""
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD: returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period, min_periods=period).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative volume weighted by price direction."""
    close = df["Close"]
    volume = df["Volume"]
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator: returns (%K, %D)."""
    low_min = df["Low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["High"].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["Close"] - low_min) / denom
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Weighted Average Price (rolling)."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = typical_price * df["Volume"]
    return tp_vol.rolling(window=period, min_periods=period).sum() / \
           df["Volume"].rolling(window=period, min_periods=period).sum()


def compute_all(df: pd.DataFrame, genome: dict) -> pd.DataFrame:
    """Compute all indicators based on genome parameters and attach to df."""
    df = df.copy()
    close = df["Close"]

    # Moving averages
    df["ma_fast"] = sma(close, int(genome["ma_fast_period"]))
    df["ma_slow"] = sma(close, int(genome["ma_slow_period"]))

    # RSI
    df["rsi"] = rsi(close, int(genome["rsi_period"]))

    # Bollinger Bands
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(
        close, int(genome["bb_period"]), genome["bb_std_dev"]
    )

    # MACD
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(
        close, int(genome["macd_fast"]), int(genome["macd_slow"]),
        int(genome["macd_signal_period"])
    )

    # ATR (volatility)
    df["atr"] = atr(df, int(genome["atr_period"]))
    df["atr_pct"] = df["atr"] / close  # ATR as % of price

    # OBV (volume trend)
    df["obv"] = obv(df)
    df["obv_sma"] = sma(df["obv"], 20)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = stochastic(
        df, int(genome["stoch_k_period"]), int(genome["stoch_d_period"])
    )

    # VWAP
    df["vwap"] = vwap(df, int(genome["vwap_period"]))

    return df
