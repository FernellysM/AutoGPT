"""
Live Trading Engine

Connects to a real crypto exchange via CCXT and executes trades
based on an evolved genome's signals.

Supports:
- Paper trading (simulated, no real money)
- Live trading (real orders on exchange)
- Safety controls (max loss kill switch, cooldowns, position limits)
- Full logging of every decision
"""

import json
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import ccxt
except ImportError:
    ccxt = None

from .agent import generate_signals
from .indicators import compute_all

logger = logging.getLogger("evo_trader.live")


class SafetyController:
    """Kill switch and safety limits for live trading."""

    def __init__(
        self,
        initial_capital: float,
        max_loss_pct: float = 0.20,        # kill if down 20% from start
        max_daily_loss_pct: float = 0.10,   # kill if down 10% in one day
        max_trades_per_day: int = 10,       # rate limit
        cooldown_after_loss: int = 300,     # 5 min cooldown after a loss
    ):
        self.initial_capital = initial_capital
        self.max_loss_pct = max_loss_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_trades_per_day = max_trades_per_day
        self.cooldown_after_loss = cooldown_after_loss

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now()
        self.last_loss_time: Optional[datetime] = None
        self.killed = False
        self.kill_reason = ""

    def check(self, current_equity: float) -> bool:
        """Returns True if trading is allowed, False if killed."""
        if self.killed:
            return False

        # Reset daily counters at midnight
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = now

        # Check total loss
        total_loss_pct = (current_equity - self.initial_capital) / self.initial_capital
        if total_loss_pct < -self.max_loss_pct:
            self.killed = True
            self.kill_reason = (f"Total loss exceeded {self.max_loss_pct:.0%}: "
                                f"equity ${current_equity:.2f} vs initial ${self.initial_capital:.2f}")
            return False

        # Check daily loss
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss_pct:
            self.killed = True
            self.kill_reason = f"Daily loss exceeded {self.max_daily_loss_pct:.0%}: ${self.daily_pnl:.2f}"
            return False

        # Check daily trade limit
        if self.daily_trades >= self.max_trades_per_day:
            logger.warning(f"Daily trade limit reached ({self.max_trades_per_day})")
            return False

        # Check cooldown after loss
        if self.last_loss_time:
            elapsed = (now - self.last_loss_time).total_seconds()
            if elapsed < self.cooldown_after_loss:
                remaining = self.cooldown_after_loss - elapsed
                logger.info(f"Cooling down after loss, {remaining:.0f}s remaining")
                return False

        return True

    def record_trade(self, pnl: float):
        self.daily_pnl += pnl
        self.daily_trades += 1
        if pnl < 0:
            self.last_loss_time = datetime.now()


class PaperExchange:
    """Simulated exchange for paper trading — no real money involved."""

    def __init__(self, initial_balance: float = 50.0, fee_pct: float = 0.001):
        self.balance = initial_balance
        self.fee_pct = fee_pct
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trade_log: list = []

    def get_balance(self) -> float:
        return self.balance

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def get_price(self, symbol: str, ohlcv_df: pd.DataFrame) -> float:
        """Get latest price from data."""
        return float(ohlcv_df["Close"].iloc[-1])

    def buy(self, symbol: str, amount_usd: float, price: float) -> Dict:
        fee = amount_usd * self.fee_pct
        net_amount = amount_usd - fee
        quantity = net_amount / price

        self.balance -= amount_usd
        self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity

        trade = {
            "time": datetime.now().isoformat(),
            "side": "buy",
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "cost": amount_usd,
            "fee": fee,
        }
        self.trade_log.append(trade)
        return trade

    def sell(self, symbol: str, quantity: float, price: float) -> Dict:
        gross = quantity * price
        fee = gross * self.fee_pct
        net = gross - fee

        self.balance += net
        self.positions[symbol] = self.positions.get(symbol, 0.0) - quantity
        if self.positions[symbol] <= 0.0001:
            del self.positions[symbol]

        trade = {
            "time": datetime.now().isoformat(),
            "side": "sell",
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "revenue": net,
            "fee": fee,
        }
        self.trade_log.append(trade)
        return trade

    def get_equity(self, prices: Dict[str, float]) -> float:
        equity = self.balance
        for symbol, qty in self.positions.items():
            if symbol in prices:
                equity += qty * prices[symbol]
        return equity


class LiveTrader:
    """
    Main live trading loop.

    Loads a genome (or ensemble of genomes), connects to an exchange,
    and trades in real-time.
    """

    def __init__(
        self,
        genome: Dict[str, float],
        symbol: str = "BTC/USDT",
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        initial_capital: float = 50.0,
        paper: bool = True,
        timeframe: str = "1h",
        lookback: int = 200,
        log_dir: str = "trade_logs",
        max_loss_pct: float = 0.20,
        max_daily_loss_pct: float = 0.10,
        ensemble_genomes: Optional[list] = None,
    ):
        self.genome = genome
        self.ensemble_genomes = ensemble_genomes or []
        self.symbol = symbol
        self.exchange_id = exchange_id
        self.initial_capital = initial_capital
        self.paper = paper
        self.timeframe = timeframe
        self.lookback = lookback
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Position tracking
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.in_position = False

        # Safety
        self.safety = SafetyController(
            initial_capital=initial_capital,
            max_loss_pct=max_loss_pct,
            max_daily_loss_pct=max_daily_loss_pct,
        )

        # Setup logging
        self._setup_logging()

        # Setup exchange
        if paper:
            logger.info("PAPER TRADING MODE — no real money")
            self.exchange = None
            self.paper_exchange = PaperExchange(initial_capital)
        else:
            if ccxt is None:
                raise ImportError("ccxt is required for live trading: pip install ccxt")
            if not api_key or not api_secret:
                raise ValueError("API key and secret required for live trading")

            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            self.paper_exchange = None
            logger.info(f"LIVE TRADING on {exchange_id} — {symbol}")

    def _setup_logging(self):
        log_file = self.log_dir / f"trader_{datetime.now():%Y%m%d_%H%M%S}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Also log to console
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(console)

    def fetch_ohlcv(self) -> pd.DataFrame:
        """Fetch recent OHLCV data from exchange or yfinance for paper mode."""
        if self.paper:
            import yfinance as yf
            # Convert symbol format: BTC/USDT -> BTC-USD
            ticker = self.symbol.split("/")[0] + "-USD"
            tf_map = {"1h": "1h", "4h": "1h", "1d": "1d", "15m": "15m"}
            interval = tf_map.get(self.timeframe, "1h")

            # Determine period based on lookback and timeframe
            if interval in ("15m", "1h"):
                period = "60d"
            else:
                period = "2y"

            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return df.tail(self.lookback)
        else:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, self.timeframe, limit=self.lookback
            )
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df

    def get_current_price(self) -> float:
        """Get the latest price."""
        if self.paper:
            df = self.fetch_ohlcv()
            return float(df["Close"].iloc[-1])
        else:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker["last"]

    def get_balance(self) -> float:
        """Get available balance in quote currency (USDT/USD)."""
        if self.paper:
            return self.paper_exchange.get_balance()
        else:
            balance = self.exchange.fetch_balance()
            quote = self.symbol.split("/")[1]
            return balance["free"].get(quote, 0.0)

    def get_position_qty(self) -> float:
        """Get current position quantity in base currency."""
        if self.paper:
            base = self.symbol.split("/")[0]
            return self.paper_exchange.get_position(base)
        else:
            balance = self.exchange.fetch_balance()
            base = self.symbol.split("/")[0]
            return balance["free"].get(base, 0.0)

    def get_equity(self) -> float:
        """Get total equity (balance + position value)."""
        price = self.get_current_price()
        if self.paper:
            base = self.symbol.split("/")[0]
            return self.paper_exchange.get_equity({base: price})
        else:
            qty = self.get_position_qty()
            return self.get_balance() + qty * price

    def execute_buy(self, amount_usd: float, price: float):
        """Execute a buy order."""
        base = self.symbol.split("/")[0]
        quantity = amount_usd / price

        logger.info(f"BUY {quantity:.6f} {base} @ ${price:,.2f} (${amount_usd:.2f})")

        if self.paper:
            trade = self.paper_exchange.buy(base, amount_usd, price)
        else:
            trade = self.exchange.create_market_buy_order(self.symbol, quantity)

        self.in_position = True
        self.entry_price = price
        self.highest_since_entry = price

        self._save_trade("BUY", price, quantity, amount_usd)
        return trade

    def execute_sell(self, quantity: float, price: float, reason: str = "signal"):
        """Execute a sell order."""
        base = self.symbol.split("/")[0]

        logger.info(f"SELL {quantity:.6f} {base} @ ${price:,.2f} (reason: {reason})")

        if self.paper:
            trade = self.paper_exchange.sell(base, quantity, price)
        else:
            trade = self.exchange.create_market_sell_order(self.symbol, quantity)

        pnl = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
        pnl_usd = quantity * (price - self.entry_price)

        logger.info(f"  P&L: {pnl:+.2%} (${pnl_usd:+.2f})")
        self.safety.record_trade(pnl_usd)

        self.in_position = False
        self.entry_price = 0.0
        self.highest_since_entry = 0.0

        self._save_trade("SELL", price, quantity, quantity * price, reason, pnl)
        return trade

    def _save_trade(self, side, price, qty, value, reason="", pnl=0.0):
        trade_record = {
            "time": datetime.now().isoformat(),
            "side": side,
            "price": price,
            "quantity": qty,
            "value": value,
            "reason": reason,
            "pnl_pct": pnl,
            "equity": self.get_equity(),
            "paper": self.paper,
        }
        log_file = self.log_dir / "trades.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(trade_record, default=str) + "\n")

    def evaluate_signal(self) -> float:
        """
        Fetch latest data and compute the current signal.
        If ensemble genomes are available, average their signals
        (majority vote — reduces variance and prevents rogue trades).
        """
        df = self.fetch_ohlcv()

        if self.ensemble_genomes:
            # Ensemble: average signal across all agents
            signals = []
            for eg in self.ensemble_genomes:
                genome = eg["genome"] if isinstance(eg, dict) else eg
                df_copy = generate_signals(df.copy(), genome)
                df_copy = df_copy.dropna()
                if len(df_copy) > 0:
                    signals.append(float(df_copy["signal_combined"].iloc[-1]))

            if not signals:
                return 0.0

            avg_signal = sum(signals) / len(signals)
            # Only trade if majority agree on direction
            agree_buy = sum(1 for s in signals if s > self.genome["signal_threshold"])
            agree_sell = sum(1 for s in signals if s < -self.genome["signal_threshold"])
            majority = len(signals) / 2

            if agree_buy > majority:
                return avg_signal
            elif agree_sell > majority:
                return avg_signal
            else:
                return 0.0  # no consensus

        else:
            df = generate_signals(df, self.genome)
            df = df.dropna()
            if len(df) == 0:
                return 0.0
            return float(df["signal_combined"].iloc[-1])

    def run_once(self):
        """Execute one trading cycle: fetch data, evaluate, act."""
        equity = self.get_equity()
        price = self.get_current_price()

        # Safety check
        if not self.safety.check(equity):
            if self.safety.killed:
                logger.critical(f"KILL SWITCH: {self.safety.kill_reason}")
                # Liquidate any open position
                if self.in_position:
                    qty = self.get_position_qty()
                    if qty > 0:
                        self.execute_sell(qty, price, reason="kill_switch")
            return False  # Stop trading

        signal = self.evaluate_signal()
        threshold = self.genome["signal_threshold"]

        logger.info(
            f"Tick | Price: ${price:,.2f} | Signal: {signal:+.3f} | "
            f"Threshold: +/-{threshold:.3f} | Equity: ${equity:.2f} | "
            f"In position: {self.in_position}"
        )

        if not self.in_position:
            # --- Entry logic ---
            if signal > threshold:
                balance = self.get_balance()
                invest = balance * self.genome["position_size_pct"]
                # Minimum order size check
                if invest < 5.0:
                    logger.warning(f"Investment too small: ${invest:.2f} (min $5)")
                    return True
                self.execute_buy(invest, price)

        else:
            # --- Exit logic ---
            self.highest_since_entry = max(self.highest_since_entry, price)
            pnl_pct = (price - self.entry_price) / self.entry_price

            hit_stop = pnl_pct <= -self.genome["stop_loss_pct"]
            hit_tp = pnl_pct >= self.genome["take_profit_pct"]
            signal_exit = signal < -threshold

            # Trailing stop
            hit_trailing = False
            trailing = self.genome["trailing_stop_pct"]
            if trailing > 0 and self.highest_since_entry > self.entry_price:
                drop = (self.highest_since_entry - price) / self.highest_since_entry
                hit_trailing = drop >= trailing

            if hit_stop or hit_tp or signal_exit or hit_trailing:
                reason = "stop_loss"
                if hit_tp:
                    reason = "take_profit"
                elif hit_trailing:
                    reason = "trailing_stop"
                elif signal_exit:
                    reason = "signal"

                qty = self.get_position_qty()
                if qty > 0:
                    self.execute_sell(qty, price, reason=reason)
            else:
                logger.info(
                    f"  Holding | P&L: {pnl_pct:+.2%} | "
                    f"Peak: ${self.highest_since_entry:,.2f} | "
                    f"SL: {-self.genome['stop_loss_pct']:.2%} | "
                    f"TP: {self.genome['take_profit_pct']:.2%}"
                )

        return True  # Continue trading

    def run(self, interval_seconds: Optional[int] = None):
        """
        Main trading loop. Runs continuously until killed or interrupted.

        Args:
            interval_seconds: Seconds between checks. If None, auto-set based on timeframe.
        """
        if interval_seconds is None:
            tf_intervals = {
                "1m": 60, "5m": 300, "15m": 900,
                "1h": 3600, "4h": 14400, "1d": 86400,
            }
            interval_seconds = tf_intervals.get(self.timeframe, 3600)

        mode_str = "PAPER" if self.paper else "LIVE"
        logger.info(f"""
{'='*60}
  EVO TRADER — {mode_str} MODE
  Symbol:    {self.symbol}
  Capital:   ${self.initial_capital:.2f}
  Timeframe: {self.timeframe}
  Interval:  {interval_seconds}s between checks
  Max loss:  {self.safety.max_loss_pct:.0%} total, {self.safety.max_daily_loss_pct:.0%} daily
{'='*60}
        """)

        try:
            while True:
                try:
                    should_continue = self.run_once()
                    if not should_continue:
                        logger.critical("Trading stopped by safety controller")
                        break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nTrading stopped by user (Ctrl+C)")

        # Final summary
        equity = self.get_equity()
        pnl = equity - self.initial_capital
        pnl_pct = pnl / self.initial_capital

        logger.info(f"""
{'='*60}
  FINAL SUMMARY
  Starting capital: ${self.initial_capital:.2f}
  Final equity:     ${equity:.2f}
  P&L:              ${pnl:+.2f} ({pnl_pct:+.2%})
  Total trades:     {self.safety.daily_trades}
{'='*60}
        """)

        if self.paper:
            trades_file = self.log_dir / "paper_trades.json"
            with open(trades_file, "w") as f:
                json.dump(self.paper_exchange.trade_log, f, indent=2, default=str)
            logger.info(f"Trade log saved to {trades_file}")
