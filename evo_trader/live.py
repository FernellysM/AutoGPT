#!/usr/bin/env python3
"""
Live/Paper Trading Entry Point

Usage:
    # Paper trading (simulated, safe — START HERE)
    python -m evo_trader.live --genome best_genome.json --paper

    # Paper trading with custom settings
    python -m evo_trader.live --genome best_genome.json --paper --symbol BTC/USDT --capital 50 --timeframe 1h

    # LIVE trading (real money — be careful!)
    python -m evo_trader.live --genome best_genome.json --live \\
        --exchange binance \\
        --api-key YOUR_KEY \\
        --api-secret YOUR_SECRET \\
        --capital 50

    # Live with environment variables (safer — don't put keys in CLI)
    export EXCHANGE_API_KEY=your_key
    export EXCHANGE_API_SECRET=your_secret
    python -m evo_trader.live --genome best_genome.json --live --capital 50
"""

import argparse
import json
import os
import sys

from .live_trader import LiveTrader


def main():
    parser = argparse.ArgumentParser(
        description="EVO TRADER — Live/Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT SAFETY NOTES:
  1. ALWAYS start with --paper mode first
  2. Only use money you can afford to lose
  3. The kill switch will stop trading if losses exceed --max-loss
  4. Use environment variables for API keys, not CLI arguments
  5. Past backtest performance does NOT guarantee future results

Examples:
  # Paper trade with your best evolved genome
  python -m evo_trader.live --genome best_genome_BTC_USD.json --paper

  # Live trade on Binance with $50
  python -m evo_trader.live --genome best_genome.json --live --capital 50
        """
    )

    # Required
    parser.add_argument("--genome", required=True,
                        help="Path to saved genome JSON file from evolution")

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true",
                      help="Paper trading (simulated, no real money)")
    mode.add_argument("--live", action="store_true",
                      help="Live trading (REAL MONEY — use with caution)")

    # Trading params
    parser.add_argument("--symbol", default="BTC/USDT",
                        help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--capital", type=float, default=50.0,
                        help="Starting capital in USD (default: 50)")
    parser.add_argument("--timeframe", default="1h",
                        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="Candle timeframe (default: 1h)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Seconds between checks (default: auto based on timeframe)")

    # Exchange
    parser.add_argument("--exchange", default="binance",
                        help="Exchange ID (default: binance). Supports: binance, coinbase, kraken, bybit")
    parser.add_argument("--api-key", default=None,
                        help="API key (or set EXCHANGE_API_KEY env var)")
    parser.add_argument("--api-secret", default=None,
                        help="API secret (or set EXCHANGE_API_SECRET env var)")

    # Safety
    parser.add_argument("--max-loss", type=float, default=0.20,
                        help="Max total loss before kill switch (default: 0.20 = 20%%)")
    parser.add_argument("--max-daily-loss", type=float, default=0.10,
                        help="Max daily loss before kill switch (default: 0.10 = 10%%)")

    args = parser.parse_args()

    # Load genome
    try:
        with open(args.genome) as f:
            data = json.load(f)
        genome = data["genome"]
        print(f"Loaded genome from: {args.genome}")
        if "training" in data:
            tr = data["training"]
            print(f"  Training return: {tr.get('total_return', 'N/A')}")
            print(f"  Training sharpe: {tr.get('sharpe_ratio', 'N/A')}")
        if "validation" in data:
            val = data["validation"]
            print(f"  Validation return: {val.get('total_return', 'N/A')}")
    except FileNotFoundError:
        print(f"ERROR: Genome file not found: {args.genome}")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: Invalid genome file: {e}")
        sys.exit(1)

    # Get API keys
    api_key = args.api_key or os.environ.get("EXCHANGE_API_KEY", "")
    api_secret = args.api_secret or os.environ.get("EXCHANGE_API_SECRET", "")

    if args.live and (not api_key or not api_secret):
        print("\nERROR: Live trading requires API keys.")
        print("Set them via --api-key/--api-secret or environment variables:")
        print("  export EXCHANGE_API_KEY=your_key")
        print("  export EXCHANGE_API_SECRET=your_secret")
        sys.exit(1)

    if args.live:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                    *** WARNING ***                        ║
║                                                          ║
║  You are about to trade with REAL MONEY.                 ║
║                                                          ║
║  Capital at risk: ${args.capital:.2f}                          ║
║  Max loss:        {args.max_loss:.0%} (${args.capital * args.max_loss:.2f})                        ║
║  Exchange:        {args.exchange:<20s}                 ║
║                                                          ║
║  Past performance does NOT guarantee future results.     ║
║  Only trade with money you can afford to lose.           ║
║                                                          ║
║  Press Ctrl+C at any time to stop.                       ║
╚══════════════════════════════════════════════════════════╝
        """)
        confirm = input("Type 'YES' to confirm: ").strip()
        if confirm != "YES":
            print("Cancelled.")
            sys.exit(0)

    # Load ensemble if available
    ensemble_genomes = data.get("ensemble", [])
    if ensemble_genomes:
        print(f"  Ensemble: {len(ensemble_genomes)} diverse agents (majority-vote trading)")
    else:
        print(f"  Single agent mode")

    # Create and run trader
    trader = LiveTrader(
        genome=genome,
        symbol=args.symbol,
        exchange_id=args.exchange,
        api_key=api_key,
        api_secret=api_secret,
        initial_capital=args.capital,
        paper=args.paper,
        timeframe=args.timeframe,
        max_loss_pct=args.max_loss,
        max_daily_loss_pct=args.max_daily_loss,
        ensemble_genomes=ensemble_genomes,
    )

    trader.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
