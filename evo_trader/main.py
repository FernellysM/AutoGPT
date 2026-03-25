#!/usr/bin/env python3
"""
Evolutionary Trading Agent - Main Entry Point

Usage:
    # Standard mode (single asset, train/test split)
    python -m evo_trader.main

    # Walk-forward mode (anti-overfit rolling windows)
    python -m evo_trader.main --mode walk_forward

    # Multi-asset mode (train across BTC, ETH, SOL, etc.)
    python -m evo_trader.main --mode multi_asset

    # Custom assets
    python -m evo_trader.main --mode multi_asset --tickers BTC-USD ETH-USD SOL-USD AVAX-USD

    # Full power run
    python -m evo_trader.main --mode multi_asset --pop 60 --gens 80 --period 3y
"""

import argparse
import json
import os
from datetime import datetime

from .backtest import fetch_data, fetch_multi_asset, split_data
from .evolution import evolve
from .genome import genome_summary


DEFAULT_MULTI_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"]


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Trading Agent")
    parser.add_argument("--ticker", default="BTC-USD", help="Asset ticker for standard mode (default: BTC-USD)")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers for multi-asset mode")
    parser.add_argument("--period", default="2y", help="Data period (default: 2y)")
    parser.add_argument("--interval", default="1d", help="Data interval (default: 1d)")
    parser.add_argument("--pop", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--gens", type=int, default=50, help="Generations (default: 50)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital (default: 10000)")
    parser.add_argument("--mutation", type=float, default=0.15, help="Mutation rate (default: 0.15)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train/val split ratio (default: 0.7)")
    parser.add_argument("--mode", choices=["standard", "walk_forward", "multi_asset", "anti_overfit"],
                        default="anti_overfit", help="Evolution mode (default: anti_overfit)")
    parser.add_argument("--wf-splits", type=int, default=5, help="Walk-forward splits (default: 5)")
    parser.add_argument("--save", default=None, help="Save best genome to JSON file")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_MULTI_TICKERS
    mode_display = {
        "standard": "Standard (single split)",
        "walk_forward": "Walk-Forward (rolling anti-overfit)",
        "multi_asset": "Multi-Asset (cross-market robustness)",
        "anti_overfit": "Anti-Overfit (val-in-loop + noise + ensemble)",
    }

    print(f"""
╔══════════════════════════════════════════════════════════╗
║         EVOLUTIONARY TRADING AGENT v2.0                  ║
║                                                          ║
║  "Survival of the most profitable"                       ║
╚══════════════════════════════════════════════════════════╝

  Mode:        {mode_display[args.mode]}
  Ticker(s):   {', '.join(tickers) if args.mode == 'multi_asset' else args.ticker}
  Period:      {args.period}
  Population:  {args.pop}
  Generations: {args.gens}
  Capital:     ${args.capital:,.2f}
  Mutation:    {args.mutation:.0%}
  Fees:        0.1% + 0.05% slippage per trade
""")

    # --- Fetch data ---
    if args.mode == "multi_asset":
        print(f"Fetching data for {len(tickers)} assets...")
        all_data = fetch_multi_asset(tickers, args.period, args.interval)
        for ticker, df in all_data.items():
            print(f"  {ticker}: {len(df)} bars, "
                  f"${df['Close'].min():,.2f} - ${df['Close'].max():,.2f}")

        # Split each asset into train/val
        train_data = {}
        val_data = {}
        for ticker, df in all_data.items():
            tr, va = split_data(df, args.train_ratio)
            train_data[ticker] = tr
            val_data[ticker] = va

    else:
        print("Fetching market data...")
        df = fetch_data(args.ticker, args.period, args.interval)
        print(f"  Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Price range: ${df['Close'].min():,.2f} - ${df['Close'].max():,.2f}")

        if args.mode == "anti_overfit":
            # 3-way split: train (50%) / val (25%) / test (25%)
            # Val is used DURING evolution, test is truly held-out
            n = len(df)
            train_data = df.iloc[:int(n * 0.50)].copy()
            val_data = df.iloc[int(n * 0.50):int(n * 0.75)].copy()
            test_data = df.iloc[int(n * 0.75):].copy()
            print(f"  Train: {len(train_data)} bars | Val (in-loop): {len(val_data)} bars | Test (held-out): {len(test_data)} bars")
            print(f"  Noise injection: ON | Overfit penalty: ON | Ensemble: ON")
        elif args.mode == "walk_forward":
            train_data = df
            _, val_data = split_data(df, args.train_ratio)
            test_data = None
            print(f"  Walk-forward: {args.wf_splits} rolling windows")
        else:
            train_data, val_data = split_data(df, args.train_ratio)
            test_data = None
            print(f"  Train: {len(train_data)} bars | Validation: {len(val_data)} bars")

    # --- Evolve ---
    print("\nStarting evolution...\n")
    best, history = evolve(
        train_data=train_data,
        val_data=val_data,
        population_size=args.pop,
        generations=args.gens,
        mutation_rate=args.mutation,
        initial_capital=args.capital,
        mode=args.mode,
        wf_splits=args.wf_splits,
    )

    # --- Final report ---
    print(f"\n{'='*60}")
    print("BEST AGENT FOUND")
    print(f"{'='*60}")
    print(f"\nGenome:")
    print(genome_summary(best.genome))
    print(f"\nPerformance:")
    print(f"  Final equity:  ${best.result['final_equity']:,.2f}")
    print(f"  Total return:  {best.result['total_return']:+.2%}")
    if "full_return" in best.result:
        print(f"  Full return:   {best.result['full_return']:+.2%} (all data)")
    print(f"  Total trades:  {best.result['total_trades']}")
    print(f"  Win rate:      {best.result['win_rate']:.1%}")
    print(f"  Avg win:       ${best.result['avg_win']:,.2f}")
    print(f"  Avg loss:      ${best.result['avg_loss']:,.2f}")
    print(f"  Sharpe ratio:  {best.result['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:  {best.result['max_drawdown']:.2%}")
    print(f"  Profit factor: {best.result.get('profit_factor', 0):.2f}")
    print(f"  Total fees:    ${best.result.get('total_fees', 0):,.2f}")

    consistency = best.result.get("consistency")
    if consistency is not None:
        print(f"  Consistency:   {consistency:.0%}")

    # Walk-forward details
    wf = best.result.get("walk_forward")
    if wf:
        print(f"\n  Walk-Forward Breakdown:")
        for i, (ret, sharpe) in enumerate(zip(wf["per_window_returns"], wf["per_window_sharpe"])):
            status = "+" if ret > 0 else "-"
            print(f"    Window {i+1}: {ret:+.2%} (Sharpe: {sharpe:.2f}) [{status}]")

    # Per-asset details
    per_asset = best.result.get("per_asset", {})
    if per_asset:
        print(f"\n  Per-Asset Breakdown:")
        for ticker, info in per_asset.items():
            status = "+" if info["return"] > 0 else "-"
            print(f"    {ticker}: {info['return']:+.2%} "
                  f"(Sharpe: {info['sharpe']:.2f}, DD: {info['drawdown']:.2%}) [{status}]")

    # Anti-overfit: show train vs val breakdown
    if args.mode == "anti_overfit":
        print(f"\n  Anti-Overfit Breakdown:")
        print(f"    Train return:  {best.result.get('train_return', 0):+.2%}")
        print(f"    Val return:    {best.result.get('val_return', 0):+.2%}")
        print(f"    Overfit gap:   {best.result.get('overfit_gap', 0):+.2%}")
        print(f"    Noise spread:  {best.result.get('noise_spread', 0):.4f}")

    # Show ensemble if available
    ensemble_data = best.result.get("ensemble", [])
    if ensemble_data and len(ensemble_data) > 1:
        print(f"\n  Ensemble ({len(ensemble_data)} diverse agents):")
        for i, e in enumerate(ensemble_data):
            print(f"    Agent {i+1}: return={e['return']:+.2%}, fitness={e['fitness']:.2f}")

    if "validation" in best.result:
        val = best.result["validation"]
        print(f"\nValidation Performance (out-of-sample):")
        print(f"  Total return:  {val['total_return']:+.2%}")
        print(f"  Total trades:  {val['total_trades']}")
        print(f"  Sharpe ratio:  {val['sharpe_ratio']:.2f}")
        print(f"  Max drawdown:  {val['max_drawdown']:.2%}")
        print(f"  Fees:          ${val.get('total_fees', 0):,.2f}")

    # Run on truly held-out test data (anti_overfit mode)
    if args.mode == "anti_overfit" and test_data is not None:
        from .backtest import evaluate_genome as _eval
        test_result = _eval(best.genome, test_data, args.capital)
        print(f"\nHELD-OUT TEST (never seen during evolution):")
        print(f"  Return:  {test_result['total_return']:+.2%}")
        print(f"  Sharpe:  {test_result['sharpe_ratio']:.2f}")
        print(f"  MaxDD:   {test_result['max_drawdown']:.2%}")
        print(f"  Trades:  {test_result['total_trades']}")
        print(f"  Fees:    ${test_result.get('total_fees', 0):,.2f}")
        if test_result['total_return'] > 0:
            print("  PASS: Strategy is profitable on truly unseen data!")
        else:
            print("  FAIL: Strategy loses on unseen data")
        best.result["held_out_test"] = test_result

    # --- Save genome ---
    ticker_label = "multi" if args.mode == "multi_asset" else args.ticker.replace("-", "_")
    save_path = args.save or f"best_genome_{ticker_label}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    output = {
        "mode": args.mode,
        "ticker": args.ticker if args.mode != "multi_asset" else tickers,
        "genome": best.genome,
        "training": {k: v for k, v in best.result.items()
                     if k not in ("equity_curve", "trades", "validation", "per_asset", "walk_forward")},
        "evolution": {
            "generations": args.gens,
            "population_size": args.pop,
            "mutation_rate": args.mutation,
            "mode": args.mode,
        },
        "created": datetime.now().isoformat(),
    }
    if "validation" in best.result:
        output["validation"] = {k: v for k, v in best.result["validation"].items()
                                if k not in ("equity_curve", "trades", "per_asset")}
    if wf:
        output["walk_forward"] = wf
    if per_asset:
        output["per_asset"] = per_asset
    if "held_out_test" in best.result:
        output["held_out_test"] = {k: v for k, v in best.result["held_out_test"].items()
                                    if k not in ("equity_curve", "trades")}
    if ensemble_data:
        output["ensemble"] = ensemble_data

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nBest genome saved to: {save_path}")


if __name__ == "__main__":
    main()
