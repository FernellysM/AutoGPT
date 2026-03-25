"""
Backtesting engine: fetches historical data and evaluates agents.

Includes:
- Multi-asset data fetching
- Walk-forward optimization (rolling train/test windows)
- Transaction cost-aware fitness scoring
- Noise injection for robustness
- Validation-in-the-loop fitness
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .agent import simulate_trades


def fetch_data(
    ticker: str = "BTC-USD",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def fetch_multi_asset(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple assets."""
    data = {}
    for ticker in tickers:
        try:
            df = fetch_data(ticker, period, interval)
            if len(df) > 50:
                data[ticker] = df
        except Exception as e:
            print(f"  Warning: failed to fetch {ticker}: {e}")
    return data


def split_data(
    df: pd.DataFrame, train_ratio: float = 0.7
) -> tuple:
    """Split data into train and validation sets to detect overfitting."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def inject_noise(df: pd.DataFrame, noise_level: float = 0.005) -> pd.DataFrame:
    """
    Add random noise to OHLCV data to prevent the strategy from memorizing
    exact price patterns. This forces the genome to learn general patterns
    rather than curve-fitting to specific historical prices.

    noise_level: standard deviation of noise as fraction of price (0.005 = 0.5%)
    """
    df = df.copy()
    n = len(df)
    for col in ["Open", "High", "Low", "Close"]:
        noise = np.random.normal(1.0, noise_level, n)
        df[col] = df[col] * noise

    # Ensure High >= Close >= Low and High >= Open >= Low
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)

    # Add volume noise too (larger, volume is noisy by nature)
    vol_noise = np.random.normal(1.0, noise_level * 5, n)
    df["Volume"] = (df["Volume"] * vol_noise).clip(lower=0)

    return df


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.6,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.

    Instead of one train/test split, creates multiple rolling windows:
    [====train====][==test==]
         [====train====][==test==]
              [====train====][==test==]

    This tests whether the strategy generalizes across different market regimes.
    """
    total_len = len(df)
    window_size = total_len // n_splits
    train_size = int(window_size * train_pct / (1 - train_pct + train_pct))

    # Minimum sizes
    min_train = max(60, int(total_len * 0.15))
    min_test = max(20, int(total_len * 0.05))

    splits = []
    step = (total_len - min_train - min_test) // max(n_splits - 1, 1)

    for i in range(n_splits):
        start = i * step
        train_end = start + min_train + int((total_len - min_train - min_test) * train_pct * (1 / n_splits))
        train_end = min(train_end, total_len - min_test)
        test_end = min(train_end + min_test + step // 2, total_len)

        if train_end - start < min_train or test_end - train_end < min_test:
            continue

        train_df = df.iloc[start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        splits.append((train_df, test_df))

    return splits


def evaluate_genome(
    genome: Dict[str, float],
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> Dict:
    """Run a backtest for a single genome on the given data."""
    return simulate_trades(df, genome, initial_capital, fee_pct, slippage_pct)


def evaluate_genome_noisy(
    genome: Dict[str, float],
    df: pd.DataFrame,
    n_noise_runs: int = 3,
    noise_level: float = 0.005,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> Dict:
    """
    Evaluate a genome on the original data PLUS multiple noisy versions.
    The fitness is the WORST result across runs — this forces the strategy
    to be robust to small price variations, not just the exact history.
    """
    results = []

    # Run on original data
    orig = simulate_trades(df, genome, initial_capital, fee_pct, slippage_pct)
    results.append(orig)

    # Run on noisy variants
    for _ in range(n_noise_runs):
        noisy_df = inject_noise(df, noise_level)
        r = simulate_trades(noisy_df, genome, initial_capital, fee_pct, slippage_pct)
        results.append(r)

    # Use WORST return and average of other metrics
    # This prevents strategies that only work on exact price sequences
    worst_idx = min(range(len(results)), key=lambda i: results[i]["total_return"])
    worst = results[worst_idx]

    avg_return = np.mean([r["total_return"] for r in results])
    min_return = min(r["total_return"] for r in results)
    return_spread = max(r["total_return"] for r in results) - min_return

    # Use a blend: 50% average + 50% worst case
    blended_return = 0.5 * avg_return + 0.5 * min_return

    result = dict(worst)  # start from worst case
    result["total_return"] = blended_return
    result["noise_min_return"] = min_return
    result["noise_avg_return"] = avg_return
    result["noise_spread"] = return_spread  # lower = more robust
    result["noise_runs"] = len(results)

    return result


def evaluate_genome_multi_asset(
    genome: Dict[str, float],
    datasets: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> Dict:
    """
    Evaluate a genome across multiple assets and return averaged metrics.
    """
    all_results = []
    for ticker, df in datasets.items():
        result = simulate_trades(df, genome, initial_capital, fee_pct, slippage_pct)
        result["ticker"] = ticker
        all_results.append(result)

    if not all_results:
        return _empty_multi_result(initial_capital)

    avg_return = np.mean([r["total_return"] for r in all_results])
    avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_results])
    worst_drawdown = min(r["max_drawdown"] for r in all_results)
    total_trades = sum(r["total_trades"] for r in all_results)
    avg_win_rate = np.mean([r["win_rate"] for r in all_results])
    total_fees = sum(r["total_fees"] for r in all_results)

    profitable_count = sum(1 for r in all_results if r["total_return"] > 0)
    consistency = profitable_count / len(all_results)
    min_return = min(r["total_return"] for r in all_results)

    return {
        "final_equity": np.mean([r["final_equity"] for r in all_results]),
        "total_return": avg_return,
        "min_return": min_return,
        "consistency": consistency,
        "total_trades": total_trades,
        "winning_trades": sum(r["winning_trades"] for r in all_results),
        "losing_trades": sum(r["losing_trades"] for r in all_results),
        "win_rate": avg_win_rate,
        "avg_win": np.mean([r["avg_win"] for r in all_results]),
        "avg_loss": np.mean([r["avg_loss"] for r in all_results]),
        "max_drawdown": worst_drawdown,
        "sharpe_ratio": avg_sharpe,
        "profit_factor": np.mean([r.get("profit_factor", 0) for r in all_results]),
        "total_fees": total_fees,
        "equity_curve": all_results[0]["equity_curve"],
        "trades": all_results[0]["trades"],
        "per_asset": {r["ticker"]: {
            "return": r["total_return"],
            "sharpe": r["sharpe_ratio"],
            "drawdown": r["max_drawdown"],
            "trades": r["total_trades"],
        } for r in all_results},
    }


def evaluate_genome_walk_forward(
    genome: Dict[str, float],
    df: pd.DataFrame,
    n_splits: int = 5,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> Dict:
    """
    Evaluate a genome using walk-forward analysis.
    Tests the strategy across multiple time windows to check consistency.
    """
    splits = walk_forward_splits(df, n_splits)
    if not splits:
        return simulate_trades(df, genome, initial_capital, fee_pct, slippage_pct)

    train_results = []
    test_results = []
    for train_df, test_df in splits:
        tr = simulate_trades(train_df, genome, initial_capital, fee_pct, slippage_pct)
        te = simulate_trades(test_df, genome, initial_capital, fee_pct, slippage_pct)
        train_results.append(tr)
        test_results.append(te)

    avg_test_return = np.mean([r["total_return"] for r in test_results])
    avg_test_sharpe = np.mean([r["sharpe_ratio"] for r in test_results])
    worst_test_dd = min(r["max_drawdown"] for r in test_results)
    total_test_trades = sum(r["total_trades"] for r in test_results)

    profitable_windows = sum(1 for r in test_results if r["total_return"] > 0)
    consistency = profitable_windows / len(test_results)

    full_result = simulate_trades(df, genome, initial_capital, fee_pct, slippage_pct)

    return {
        "final_equity": full_result["final_equity"],
        "total_return": avg_test_return,
        "full_return": full_result["total_return"],
        "consistency": consistency,
        "total_trades": total_test_trades,
        "winning_trades": sum(r["winning_trades"] for r in test_results),
        "losing_trades": sum(r["losing_trades"] for r in test_results),
        "win_rate": np.mean([r["win_rate"] for r in test_results]),
        "avg_win": np.mean([r["avg_win"] for r in test_results]),
        "avg_loss": np.mean([r["avg_loss"] for r in test_results]),
        "max_drawdown": worst_test_dd,
        "sharpe_ratio": avg_test_sharpe,
        "profit_factor": np.mean([r.get("profit_factor", 0) for r in test_results]),
        "total_fees": sum(r["total_fees"] for r in test_results),
        "equity_curve": full_result["equity_curve"],
        "trades": full_result["trades"],
        "walk_forward": {
            "n_splits": len(splits),
            "profitable_windows": profitable_windows,
            "per_window_returns": [r["total_return"] for r in test_results],
            "per_window_sharpe": [r["sharpe_ratio"] for r in test_results],
        },
    }


def evaluate_with_validation(
    genome: Dict[str, float],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_weight: float = 0.4,
    val_weight: float = 0.6,
    use_noise: bool = True,
    initial_capital: float = 10000.0,
) -> Dict:
    """
    Evaluate a genome on BOTH train and validation data, returning a combined result.
    This is the key anti-overfit mechanism: the evolution directly optimizes for
    validation performance, not just training performance.

    The val_weight > train_weight means we care MORE about unseen data.
    """
    if use_noise:
        train_result = evaluate_genome_noisy(genome, train_df, n_noise_runs=2,
                                              initial_capital=initial_capital)
    else:
        train_result = simulate_trades(train_df, genome, initial_capital)

    val_result = simulate_trades(val_df, genome, initial_capital)

    # Blend returns: favor validation performance
    blended_return = (train_weight * train_result["total_return"] +
                      val_weight * val_result["total_return"])
    blended_sharpe = (train_weight * train_result["sharpe_ratio"] +
                      val_weight * val_result["sharpe_ratio"])

    # Overfit penalty: if train >> val, the strategy is memorizing
    overfit_gap = train_result["total_return"] - val_result["total_return"]
    overfit_penalty = max(0, overfit_gap) * 0.5  # penalize large gaps

    # Combine results
    result = {
        "final_equity": val_result["final_equity"],
        "total_return": blended_return - overfit_penalty,
        "train_return": train_result["total_return"],
        "val_return": val_result["total_return"],
        "overfit_gap": overfit_gap,
        "total_trades": train_result["total_trades"] + val_result["total_trades"],
        "winning_trades": train_result["winning_trades"] + val_result["winning_trades"],
        "losing_trades": train_result["losing_trades"] + val_result["losing_trades"],
        "win_rate": (train_result["win_rate"] + val_result["win_rate"]) / 2,
        "avg_win": (train_result["avg_win"] + val_result["avg_win"]) / 2,
        "avg_loss": (train_result["avg_loss"] + val_result["avg_loss"]) / 2,
        "max_drawdown": min(train_result["max_drawdown"], val_result["max_drawdown"]),
        "sharpe_ratio": blended_sharpe,
        "profit_factor": (train_result.get("profit_factor", 0) + val_result.get("profit_factor", 0)) / 2,
        "total_fees": train_result.get("total_fees", 0) + val_result.get("total_fees", 0),
        "equity_curve": train_result["equity_curve"],
        "trades": train_result["trades"],
        "noise_spread": train_result.get("noise_spread", 0),
    }

    return result


def fitness_score(result: Dict, mode: str = "standard") -> float:
    """
    Compute a fitness score from backtest results.

    Modes:
    - "standard": basic fitness
    - "robust": penalizes inconsistency, rewards generalization
    - "anti_overfit": heavily penalizes train/val divergence
    """
    ret = result["total_return"]
    sharpe = result["sharpe_ratio"]
    drawdown = abs(result["max_drawdown"])
    n_trades = result["total_trades"]
    consistency = result.get("consistency", 1.0)

    # Kill condition: negative return = heavily penalized
    if ret < 0:
        return ret * 2

    # Base score
    score = (
        ret * 100
        + sharpe * 10
        - drawdown * 50
        + min(n_trades, 50) * 0.5
    )

    if mode in ("robust", "anti_overfit"):
        # Bonus for consistency across windows/assets
        score *= (0.5 + 0.5 * consistency)

        # Penalize strategies with very few trades (might just be lucky)
        if n_trades < 5:
            score *= 0.5

        # Profit factor bonus
        pf = result.get("profit_factor", 0)
        if pf > 1:
            score += min(pf, 5) * 5

        # Fee awareness
        fees = result.get("total_fees", 0)
        initial = result.get("final_equity", 10000) - (result["total_return"] * 10000)
        if initial > 0:
            fee_ratio = fees / initial
            score -= fee_ratio * 100

    if mode == "anti_overfit":
        # Penalize overfit gap
        overfit_gap = result.get("overfit_gap", 0)
        score -= abs(overfit_gap) * 200  # heavy penalty for divergence

        # Penalize noise sensitivity
        noise_spread = result.get("noise_spread", 0)
        score -= noise_spread * 100  # penalize strategies fragile to small changes

        # Bonus for positive validation return
        val_return = result.get("val_return", ret)
        if val_return > 0:
            score += val_return * 50
        else:
            score += val_return * 150  # extra punishment for val loss

    return score


def _empty_multi_result(initial_capital: float) -> Dict:
    return {
        "final_equity": initial_capital,
        "total_return": 0.0,
        "min_return": 0.0,
        "consistency": 0.0,
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
        "per_asset": {},
    }
