"""
Trading Agent: takes a genome and produces buy/sell signals on price data.

The agent scores each bar using weighted signals from indicators,
then enters/exits positions based on the combined score and risk rules.

Includes:
- 7 technical indicator signals (MA, RSI, BB, MACD, Stochastic, OBV, VWAP)
- ATR-based volatility filter
- Transaction cost modeling (fees + slippage)
- Trailing stop loss
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .indicators import compute_all


# Default trading costs
DEFAULT_FEE_PCT = 0.001       # 0.1% per trade (typical crypto exchange)
DEFAULT_SLIPPAGE_PCT = 0.0005  # 0.05% slippage estimate


def generate_signals(df: pd.DataFrame, genome: Dict[str, float]) -> pd.DataFrame:
    """
    Generate a signal score for each bar.

    Returns df with added columns for each indicator signal and the combined score.
    """
    df = compute_all(df, genome)

    # --- MA crossover signal ---
    df["signal_ma"] = 0.0
    df.loc[df["ma_fast"] > df["ma_slow"], "signal_ma"] = 1.0
    df.loc[df["ma_fast"] < df["ma_slow"], "signal_ma"] = -1.0

    # --- RSI signal ---
    df["signal_rsi"] = 0.0
    df.loc[df["rsi"] < genome["rsi_oversold"], "signal_rsi"] = 1.0
    df.loc[df["rsi"] > genome["rsi_overbought"], "signal_rsi"] = -1.0
    mid = (genome["rsi_oversold"] + genome["rsi_overbought"]) / 2
    neutral = df["signal_rsi"] == 0.0
    df.loc[neutral, "signal_rsi"] = (mid - df.loc[neutral, "rsi"]) / (mid - genome["rsi_oversold"])
    df["signal_rsi"] = df["signal_rsi"].clip(-1, 1)

    # --- Bollinger Band signal ---
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    bb_position = (df["Close"] - df["bb_lower"]) / bb_range
    df["signal_bb"] = (1 - 2 * bb_position).clip(-1, 1)

    # --- MACD signal ---
    # Positive histogram = bullish, negative = bearish, normalized by ATR
    macd_norm = df["atr"].replace(0, np.nan)
    df["signal_macd"] = (df["macd_hist"] / macd_norm).clip(-1, 1)

    # --- Stochastic signal ---
    df["signal_stoch"] = 0.0
    df.loc[df["stoch_k"] < genome["stoch_oversold"], "signal_stoch"] = 1.0
    df.loc[df["stoch_k"] > genome["stoch_overbought"], "signal_stoch"] = -1.0
    # K crossing above D is bullish
    stoch_cross = df["stoch_k"] - df["stoch_d"]
    stoch_neutral = df["signal_stoch"] == 0.0
    df.loc[stoch_neutral, "signal_stoch"] = (stoch_cross.loc[stoch_neutral] / 50).clip(-1, 1)

    # --- OBV signal ---
    # OBV above its SMA = accumulation (bullish), below = distribution
    obv_diff = df["obv"] - df["obv_sma"]
    obv_scale = df["obv_sma"].abs().replace(0, np.nan)
    df["signal_obv"] = (obv_diff / obv_scale).clip(-1, 1)

    # --- VWAP signal ---
    # Price above VWAP = bullish, below = bearish
    vwap_diff = df["Close"] - df["vwap"]
    df["signal_vwap"] = (vwap_diff / df["atr"].replace(0, np.nan)).clip(-1, 1)

    # --- Combined weighted signal ---
    weights = {
        "signal_ma": genome["weight_ma_cross"],
        "signal_rsi": genome["weight_rsi"],
        "signal_bb": genome["weight_bb"],
        "signal_macd": genome["weight_macd"],
        "signal_stoch": genome["weight_stoch"],
        "signal_obv": genome["weight_obv"],
        "signal_vwap": genome["weight_vwap"],
    }
    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1.0

    df["signal_combined"] = sum(
        w * df[sig] for sig, w in weights.items()
    ) / total_weight

    # --- Volatility filter ---
    # Suppress signals when volatility is too low (market is flat/dead)
    vol_mask = df["atr_pct"] < genome["atr_volatility_threshold"]
    df.loc[vol_mask, "signal_combined"] = 0.0

    return df


def simulate_trades(df: pd.DataFrame, genome: Dict[str, float],
                    initial_capital: float = 10000.0,
                    fee_pct: float = DEFAULT_FEE_PCT,
                    slippage_pct: float = DEFAULT_SLIPPAGE_PCT) -> Dict:
    """
    Simulate trading based on signals with transaction costs.

    Rules:
    - Enter long when signal_combined > signal_threshold
    - Exit when signal_combined < -signal_threshold OR stop_loss/take_profit/trailing_stop hit
    - Only one position at a time
    - Each trade incurs fees + slippage on both entry and exit
    """
    df = generate_signals(df, genome)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 10:
        return _empty_result(initial_capital)

    capital = initial_capital
    position = 0.0        # units held
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    trades: List[Dict] = []
    equity_curve = [capital]
    threshold = genome["signal_threshold"]
    trailing_stop = genome["trailing_stop_pct"]
    total_fees = 0.0

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        signal = df["signal_combined"].iloc[i]

        if position == 0:
            # --- Check for entry ---
            if signal > threshold:
                invest = capital * genome["position_size_pct"]
                # Apply entry costs
                entry_cost = invest * (fee_pct + slippage_pct)
                total_fees += entry_cost
                effective_invest = invest - entry_cost
                position = effective_invest / price
                entry_price = price
                highest_since_entry = price
                capital -= invest

        else:
            # Track highest price since entry (for trailing stop)
            highest_since_entry = max(highest_since_entry, price)

            # --- Check for exit ---
            pnl_pct = (price - entry_price) / entry_price
            hit_stop = pnl_pct <= -genome["stop_loss_pct"]
            hit_tp = pnl_pct >= genome["take_profit_pct"]
            signal_exit = signal < -threshold

            # Trailing stop: if price drops X% from its peak since entry
            hit_trailing = False
            if trailing_stop > 0 and highest_since_entry > entry_price:
                drop_from_peak = (highest_since_entry - price) / highest_since_entry
                hit_trailing = drop_from_peak >= trailing_stop

            if hit_stop or hit_tp or signal_exit or hit_trailing:
                gross_value = position * price
                # Apply exit costs
                exit_cost = gross_value * (fee_pct + slippage_pct)
                total_fees += exit_cost
                sell_value = gross_value - exit_cost
                profit = sell_value - (position * entry_price)
                capital += sell_value

                reason = "stop_loss"
                if hit_tp:
                    reason = "take_profit"
                elif hit_trailing:
                    reason = "trailing_stop"
                elif signal_exit:
                    reason = "signal"

                trades.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": profit,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                })
                position = 0.0
                entry_price = 0.0
                highest_since_entry = 0.0

        # Track equity
        current_equity = capital + position * price
        equity_curve.append(current_equity)

    # Close any open position at end
    if position > 0:
        final_price = df["Close"].iloc[-1]
        gross_value = position * final_price
        exit_cost = gross_value * (fee_pct + slippage_pct)
        total_fees += exit_cost
        capital += gross_value - exit_cost
        position = 0.0

    final_equity = capital
    total_return = (final_equity - initial_capital) / initial_capital

    # Compute max drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]

    avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0

    # Sharpe ratio (annualized, daily returns)
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    else:
        sharpe = 0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (100.0 if gross_profit > 0 else 0.0)

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / max(len(trades), 1),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "total_fees": total_fees,
        "equity_curve": equity_curve,
        "trades": trades,
    }


def _empty_result(initial_capital: float) -> Dict:
    return {
        "final_equity": initial_capital,
        "total_return": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "profit_factor": 0.0,
        "total_fees": 0.0,
        "equity_curve": [initial_capital],
        "trades": [],
    }
