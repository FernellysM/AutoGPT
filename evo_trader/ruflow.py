"""
Ruflow — Ruflo-style Multi-Agent Swarm for Evo Trader

Inspired by the Ruflo / RuFlow multi-agent orchestration architecture
(github.com/ruvnet/ruflo), this module decomposes the trading workflow
into a swarm of specialised agents that execute in parallel and share a
common context store.

Agent roles
-----------
  MarketAnalystAgent  — fetches OHLCV data, detects market regime
  EvolverAgent        — runs the genetic algorithm to find an optimal genome
  RiskManagerAgent    — audits the evolved genome's risk profile
  ReporterAgent       — assembles the full pipeline report

Swarm flow
----------
  1. MarketAnalystAgent  ──► SharedContext (data, regime)
  2. EvolverAgent        ──► SharedContext (genome, fitness, history)   [uses data]
  3. RiskManagerAgent    ──► SharedContext (risk_report)                [uses genome + data]
  4. ReporterAgent       ──► SharedContext (final_report)               [uses everything]

Usage
-----
    python -m evo_trader.main --mode ruflow
    python -m evo_trader.main --mode ruflow --ticker ETH-USD --pop 40 --gens 60

    # Programmatic
    from evo_trader.ruflow import RuflowSwarm, SwarmConfig
    swarm = RuflowSwarm(SwarmConfig(ticker="BTC-USD", pop_size=30, gens=50))
    report = swarm.run()
"""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import (
    evaluate_genome,
    evaluate_genome_walk_forward,
    fetch_data,
    fetch_multi_asset,
    split_data,
)
from .evolution import evolve, Individual
from .genome import genome_summary


# ---------------------------------------------------------------------------
# Shared Context (the "memory" that all agents read/write)
# ---------------------------------------------------------------------------

class SharedContext:
    """Thread-safe key-value store shared across all swarm agents."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._events: Dict[str, threading.Event] = {}

    # ---- basic access ----

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            if key in self._events:
                self._events[key].set()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def wait_for(self, key: str, timeout: float = 600.0) -> bool:
        """Block until `key` is written into the context (or timeout)."""
        with self._lock:
            if key in self._data:
                return True
            if key not in self._events:
                self._events[key] = threading.Event()
            ev = self._events[key]
        return ev.wait(timeout=timeout)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Abstract Ruflo agent.

    Each agent has a `role` (string label), receives the shared context,
    and implements `execute()` which writes its outputs back to the context.
    """

    role: str = "base"

    def __init__(self, ctx: SharedContext) -> None:
        self.ctx = ctx
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.error: Optional[Exception] = None

    def execute(self) -> None:  # noqa: D102
        raise NotImplementedError

    def run(self) -> None:
        """Wrapper that records timing and catches errors."""
        self.started_at = datetime.now()
        try:
            self.execute()
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            self.ctx.set(f"{self.role}_error", str(exc))
        finally:
            self.finished_at = datetime.now()

    @property
    def elapsed(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0


# ---------------------------------------------------------------------------
# Agent 1: Market Analyst
# ---------------------------------------------------------------------------

class MarketAnalystAgent(BaseAgent):
    """
    Fetches OHLCV data for the target ticker(s) and detects the current
    market regime (trending_up / trending_down / ranging / volatile).

    Writes to context
    -----------------
      market_data      — pd.DataFrame  (single) or Dict[str, DataFrame] (multi)
      train_data       — pd.DataFrame or dict
      val_data         — pd.DataFrame or dict
      market_regime    — str
      regime_details   — dict
      tickers          — List[str]
    """

    role = "market_analyst"

    def __init__(
        self,
        ctx: SharedContext,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        train_ratio: float = 0.70,
    ) -> None:
        super().__init__(ctx)
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.train_ratio = train_ratio

    # ---- regime detection ----

    @staticmethod
    def _detect_regime(df: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Classify recent price action into one of four regimes using
        simple heuristics on the last 90 bars (or all bars if shorter).
        """
        tail = df.tail(min(90, len(df))).copy()
        close = tail["Close"].values

        # Trend: slope of linear regression
        x = np.arange(len(close))
        slope = float(np.polyfit(x, close, 1)[0])
        slope_pct = slope / close[0] if close[0] > 0 else 0

        # Volatility: daily returns std
        returns = pd.Series(close).pct_change().dropna()
        vol = float(returns.std())

        # Range: high/low ratio over the window
        hl_ratio = float((tail["High"].max() - tail["Low"].min()) / tail["Close"].mean())

        if vol > 0.04:
            regime = "volatile"
        elif slope_pct > 0.003:
            regime = "trending_up"
        elif slope_pct < -0.003:
            regime = "trending_down"
        else:
            regime = "ranging"

        details = {
            "slope_pct_per_bar": round(slope_pct, 6),
            "daily_vol": round(vol, 4),
            "hl_ratio": round(hl_ratio, 4),
        }
        return regime, details

    def execute(self) -> None:
        print(f"\n[{self.role}] Fetching data for {self.tickers} | "
              f"period={self.period} interval={self.interval}")

        multi = len(self.tickers) > 1

        if multi:
            all_data = fetch_multi_asset(self.tickers, self.period, self.interval)
            if not all_data:
                raise RuntimeError("No data fetched for any ticker")

            train_data, val_data = {}, {}
            for ticker, df in all_data.items():
                tr, va = split_data(df, self.train_ratio)
                train_data[ticker] = tr
                val_data[ticker] = va
                print(f"  {ticker}: {len(df)} bars "
                      f"${df['Close'].min():,.2f}–${df['Close'].max():,.2f}")

            # Detect regime on the first ticker
            first_df = next(iter(all_data.values()))
            regime, regime_details = self._detect_regime(first_df)
            market_data = all_data

        else:
            df = fetch_data(self.tickers[0], self.period, self.interval)
            print(f"  {self.tickers[0]}: {len(df)} bars "
                  f"${df['Close'].min():,.2f}–${df['Close'].max():,.2f}")
            train_data, val_data = split_data(df, self.train_ratio)
            regime, regime_details = self._detect_regime(df)
            market_data = df

        print(f"  Market regime: {regime} {regime_details}")

        self.ctx.set("market_data", market_data)
        self.ctx.set("train_data", train_data)
        self.ctx.set("val_data", val_data)
        self.ctx.set("market_regime", regime)
        self.ctx.set("regime_details", regime_details)
        self.ctx.set("tickers", self.tickers)
        self.ctx.set("is_multi_asset", multi)


# ---------------------------------------------------------------------------
# Agent 2: Evolver
# ---------------------------------------------------------------------------

class EvolverAgent(BaseAgent):
    """
    Waits for market data from the analyst, then runs the genetic algorithm
    to find the best trading genome.

    Writes to context
    -----------------
      best_individual  — Individual
      evolution_history — List[Dict]
      evolution_mode    — str  (auto-selected based on regime)
    """

    role = "evolver"

    def __init__(
        self,
        ctx: SharedContext,
        pop_size: int = 30,
        gens: int = 50,
        mutation_rate: float = 0.15,
        initial_capital: float = 10_000.0,
        force_mode: Optional[str] = None,
        wf_splits: int = 5,
    ) -> None:
        super().__init__(ctx)
        self.pop_size = pop_size
        self.gens = gens
        self.mutation_rate = mutation_rate
        self.initial_capital = initial_capital
        self.force_mode = force_mode
        self.wf_splits = wf_splits

    @staticmethod
    def _select_mode(regime: str) -> str:
        """Pick the evolution mode best suited to the detected regime."""
        return {
            "trending_up": "anti_overfit",
            "trending_down": "walk_forward",
            "ranging": "anti_overfit",
            "volatile": "walk_forward",
        }.get(regime, "anti_overfit")

    def execute(self) -> None:
        # Wait for the analyst to finish
        if not self.ctx.wait_for("train_data", timeout=600):
            raise RuntimeError("Timed out waiting for market data from analyst")

        train_data = self.ctx.get("train_data")
        val_data = self.ctx.get("val_data")
        regime = self.ctx.get("market_regime", "ranging")
        is_multi = self.ctx.get("is_multi_asset", False)

        mode = self.force_mode or (
            "multi_asset" if is_multi else self._select_mode(regime)
        )
        self.ctx.set("evolution_mode", mode)

        print(f"\n[{self.role}] Evolving | mode={mode} pop={self.pop_size} "
              f"gens={self.gens} regime={regime}")

        best, history = evolve(
            train_data=train_data,
            val_data=val_data,
            population_size=self.pop_size,
            generations=self.gens,
            mutation_rate=self.mutation_rate,
            initial_capital=self.initial_capital,
            mode=mode,
            wf_splits=self.wf_splits,
            verbose=True,
        )

        self.ctx.set("best_individual", best)
        self.ctx.set("evolution_history", history)

        print(f"\n[{self.role}] Best genome found | "
              f"fitness={best.fitness:.4f} "
              f"return={best.result.get('total_return', 0):+.2%}")


# ---------------------------------------------------------------------------
# Agent 3: Risk Manager
# ---------------------------------------------------------------------------

class RiskManagerAgent(BaseAgent):
    """
    Waits for the evolver, then audits the best genome's risk profile
    against configurable thresholds.

    Writes to context
    -----------------
      risk_report    — Dict with pass/fail flags and details
      risk_approved  — bool
    """

    role = "risk_manager"

    def __init__(
        self,
        ctx: SharedContext,
        initial_capital: float = 10_000.0,
        max_drawdown_limit: float = -0.30,
        min_sharpe: float = 0.0,
        min_win_rate: float = 0.35,
        min_trades: int = 5,
    ) -> None:
        super().__init__(ctx)
        self.initial_capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades

    def execute(self) -> None:
        if not self.ctx.wait_for("best_individual", timeout=7200):
            raise RuntimeError("Timed out waiting for genome from evolver")

        best: Individual = self.ctx.get("best_individual")
        market_data = self.ctx.get("market_data")
        is_multi = self.ctx.get("is_multi_asset", False)

        # Run a full backtest on all available data for the risk audit
        if is_multi and isinstance(market_data, dict):
            # Use the first ticker for the risk evaluation pass
            eval_df = next(iter(market_data.values()))
        else:
            eval_df = market_data

        result = evaluate_genome(best.genome, eval_df, self.initial_capital)

        checks: Dict[str, Dict] = {}

        # 1. Drawdown check
        dd = result.get("max_drawdown", 0)
        checks["drawdown"] = {
            "value": round(dd, 4),
            "limit": self.max_drawdown_limit,
            "passed": dd >= self.max_drawdown_limit,
        }

        # 2. Sharpe check
        sharpe = result.get("sharpe_ratio", 0)
        checks["sharpe"] = {
            "value": round(sharpe, 4),
            "limit": self.min_sharpe,
            "passed": sharpe >= self.min_sharpe,
        }

        # 3. Win-rate check
        win_rate = result.get("win_rate", 0)
        checks["win_rate"] = {
            "value": round(win_rate, 4),
            "limit": self.min_win_rate,
            "passed": win_rate >= self.min_win_rate,
        }

        # 4. Trade count (avoid strategies that never trade)
        trades = result.get("total_trades", 0)
        checks["trade_count"] = {
            "value": trades,
            "limit": self.min_trades,
            "passed": trades >= self.min_trades,
        }

        approved = all(c["passed"] for c in checks.items() if isinstance(c, dict))
        approved = all(v["passed"] for v in checks.values())

        risk_report = {
            "approved": approved,
            "checks": checks,
            "full_backtest": {
                k: v for k, v in result.items()
                if k not in ("equity_curve", "trades")
            },
        }

        self.ctx.set("risk_report", risk_report)
        self.ctx.set("risk_approved", approved)

        status = "APPROVED" if approved else "REJECTED"
        print(f"\n[{self.role}] Risk audit {status}")
        for name, check in checks.items():
            flag = "PASS" if check["passed"] else "FAIL"
            print(f"  [{flag}] {name}: {check['value']} "
                  f"(limit: {check['limit']})")


# ---------------------------------------------------------------------------
# Agent 4: Reporter
# ---------------------------------------------------------------------------

class ReporterAgent(BaseAgent):
    """
    Waits for all other agents, then assembles and prints the full
    pipeline report, and saves JSON artefacts.

    Writes to context
    -----------------
      final_report  — Dict
    """

    role = "reporter"

    def __init__(
        self,
        ctx: SharedContext,
        save_dir: str = "ruflow_output",
    ) -> None:
        super().__init__(ctx)
        self.save_dir = Path(save_dir)

    def execute(self) -> None:
        for key in ("best_individual", "risk_report"):
            if not self.ctx.wait_for(key, timeout=7200):
                raise RuntimeError(f"Timed out waiting for {key}")

        best: Individual = self.ctx.get("best_individual")
        risk: Dict = self.ctx.get("risk_report")
        history: List[Dict] = self.ctx.get("evolution_history", [])
        regime: str = self.ctx.get("market_regime", "unknown")
        regime_details: Dict = self.ctx.get("regime_details", {})
        mode: str = self.ctx.get("evolution_mode", "unknown")
        tickers: List[str] = self.ctx.get("tickers", [])

        report = {
            "created_at": datetime.now().isoformat(),
            "tickers": tickers,
            "market_regime": regime,
            "regime_details": regime_details,
            "evolution_mode": mode,
            "evolution_summary": {
                "generations": len(history),
                "best_fitness": best.fitness,
                "best_return": best.result.get("total_return", 0),
                "val_return": best.result.get("val_return", 0),
                "sharpe_ratio": best.result.get("sharpe_ratio", 0),
                "max_drawdown": best.result.get("max_drawdown", 0),
                "total_trades": best.result.get("total_trades", 0),
                "win_rate": best.result.get("win_rate", 0),
                "profit_factor": best.result.get("profit_factor", 0),
                "total_fees": best.result.get("total_fees", 0),
            },
            "risk": risk,
            "genome": best.genome,
            "ensemble": best.result.get("ensemble", []),
        }

        # Anti-overfit breakdown
        if "train_return" in best.result:
            report["anti_overfit"] = {
                "train_return": best.result.get("train_return", 0),
                "val_return": best.result.get("val_return", 0),
                "overfit_gap": best.result.get("overfit_gap", 0),
                "noise_spread": best.result.get("noise_spread", 0),
            }

        self.ctx.set("final_report", report)

        # Save artefacts
        self.save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_label = "_".join(tickers).replace("/", "-").replace("-USD", "")

        report_path = self.save_dir / f"ruflow_{ticker_label}_{ts}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self._print_report(report, genome_summary(best.genome))
        print(f"\n[{self.role}] Report saved → {report_path}")

    @staticmethod
    def _print_report(report: Dict, genome_str: str) -> None:
        ev = report["evolution_summary"]
        risk = report["risk"]
        status = "APPROVED" if risk["approved"] else "REJECTED"

        print(f"""
{'='*64}
  RUFLOW SWARM — FINAL REPORT
{'='*64}
  Ticker(s):      {', '.join(report['tickers'])}
  Market regime:  {report['market_regime']}  {report['regime_details']}
  Evolution mode: {report['evolution_mode']}
  Generations:    {ev['generations']}

  Performance
  -----------
  Best return:    {ev['best_return']:+.2%}
  Val return:     {ev['val_return']:+.2%}
  Sharpe ratio:   {ev['sharpe_ratio']:.2f}
  Max drawdown:   {ev['max_drawdown']:.2%}
  Win rate:       {ev['win_rate']:.1%}
  Total trades:   {ev['total_trades']}
  Profit factor:  {ev['profit_factor']:.2f}
  Fees paid:      ${ev['total_fees']:,.2f}

  Risk Audit:     {status}""")

        for name, check in risk["checks"].items():
            flag = "PASS" if check["passed"] else "FAIL"
            print(f"    [{flag}] {name}: {check['value']} (limit: {check['limit']})")

        if "anti_overfit" in report:
            ao = report["anti_overfit"]
            print(f"""
  Anti-Overfit
  ------------
  Train:          {ao['train_return']:+.2%}
  Val:            {ao['val_return']:+.2%}
  Gap:            {ao['overfit_gap']:+.2%}
  Noise spread:   {ao['noise_spread']:.4f}""")

        print(f"""
  Genome
  ------
{genome_str}
{'='*64}""")


# ---------------------------------------------------------------------------
# Swarm Configuration
# ---------------------------------------------------------------------------

@dataclass
class SwarmConfig:
    """All parameters for a Ruflow swarm run."""

    # Market Analyst
    tickers: List[str] = field(default_factory=lambda: ["BTC-USD"])
    period: str = "2y"
    interval: str = "1d"
    train_ratio: float = 0.70

    # Evolver
    pop_size: int = 30
    gens: int = 50
    mutation_rate: float = 0.15
    initial_capital: float = 10_000.0
    force_mode: Optional[str] = None   # None = auto-select by regime
    wf_splits: int = 5

    # Risk Manager
    max_drawdown_limit: float = -0.30
    min_sharpe: float = 0.0
    min_win_rate: float = 0.35
    min_trades: int = 5

    # Reporter
    save_dir: str = "ruflow_output"


# ---------------------------------------------------------------------------
# Swarm Orchestrator
# ---------------------------------------------------------------------------

class RuflowSwarm:
    """
    Orchestrates the four-agent pipeline.

    Execution order (sequential within each stage, with shared context):

      Stage 1 (sequential prerequisite):
        MarketAnalystAgent  — must finish before Evolver can start

      Stage 2 (blocked on context keys, run in threads):
        EvolverAgent        — starts after train_data is ready
        (RiskManagerAgent waits for best_individual)
        (ReporterAgent waits for best_individual + risk_report)

    In practice the current default topology is fully sequential because
    the Evolver blocks on market data and all downstream agents block on
    the Evolver. Running them in threads allows the swarm to be extended
    with truly parallel agents (e.g., running multiple EvolverAgents on
    different tickers simultaneously) without changing the orchestrator.
    """

    def __init__(self, cfg: SwarmConfig) -> None:
        self.cfg = cfg
        self.ctx = SharedContext()

        self.analyst = MarketAnalystAgent(
            ctx=self.ctx,
            tickers=cfg.tickers,
            period=cfg.period,
            interval=cfg.interval,
            train_ratio=cfg.train_ratio,
        )
        self.evolver = EvolverAgent(
            ctx=self.ctx,
            pop_size=cfg.pop_size,
            gens=cfg.gens,
            mutation_rate=cfg.mutation_rate,
            initial_capital=cfg.initial_capital,
            force_mode=cfg.force_mode,
            wf_splits=cfg.wf_splits,
        )
        self.risk_manager = RiskManagerAgent(
            ctx=self.ctx,
            initial_capital=cfg.initial_capital,
            max_drawdown_limit=cfg.max_drawdown_limit,
            min_sharpe=cfg.min_sharpe,
            min_win_rate=cfg.min_win_rate,
            min_trades=cfg.min_trades,
        )
        self.reporter = ReporterAgent(
            ctx=self.ctx,
            save_dir=cfg.save_dir,
        )

        self._agents: List[BaseAgent] = [
            self.analyst,
            self.evolver,
            self.risk_manager,
            self.reporter,
        ]

    def run(self) -> Dict:
        """
        Run the full swarm pipeline.

        Analyst runs synchronously first (other agents depend on its output).
        The remaining three agents run in separate threads so that any truly
        parallel work (e.g., multiple asset analysers) can be added later
        without changing this orchestrator.

        Returns the final report dict from the Reporter agent.
        """
        print(f"""
{'='*64}
  RUFLOW SWARM  —  Multi-Agent Evo Trader
  Tickers:  {', '.join(self.cfg.tickers)}
  Agents:   {len(self._agents)} (MarketAnalyst → Evolver → RiskManager → Reporter)
{'='*64}""")

        start = time.time()

        # Stage 1: Analyst (must complete before anything else can proceed)
        self.analyst.run()
        if self.analyst.error:
            raise RuntimeError(f"MarketAnalystAgent failed: {self.analyst.error}")

        # Stage 2: Evolver, RiskManager, Reporter — each blocks on context keys
        threads = [
            threading.Thread(target=agent.run, name=agent.role, daemon=True)
            for agent in [self.evolver, self.risk_manager, self.reporter]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=14_400)  # 4-hour hard cap

        # Collect errors
        errors = [
            f"{agent.role}: {agent.error}"
            for agent in self._agents
            if agent.error is not None
        ]
        if errors:
            raise RuntimeError("Swarm agent failures:\n" + "\n".join(errors))

        elapsed = time.time() - start
        print(f"\nSwarm completed in {elapsed:.1f}s")
        self._print_timing()

        return self.ctx.get("final_report", {})

    def _print_timing(self) -> None:
        print("\nAgent timings:")
        for agent in self._agents:
            status = "ERROR" if agent.error else "OK"
            print(f"  {agent.role:<20} {agent.elapsed:6.1f}s  [{status}]")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_ruflow(
    tickers: Optional[List[str]] = None,
    period: str = "2y",
    interval: str = "1d",
    pop_size: int = 30,
    gens: int = 50,
    mutation_rate: float = 0.15,
    initial_capital: float = 10_000.0,
    force_mode: Optional[str] = None,
    wf_splits: int = 5,
    save_dir: str = "ruflow_output",
) -> Dict:
    """Single-call entry point for the Ruflow swarm."""
    cfg = SwarmConfig(
        tickers=tickers or ["BTC-USD"],
        period=period,
        interval=interval,
        pop_size=pop_size,
        gens=gens,
        mutation_rate=mutation_rate,
        initial_capital=initial_capital,
        force_mode=force_mode,
        wf_splits=wf_splits,
        save_dir=save_dir,
    )
    swarm = RuflowSwarm(cfg)
    return swarm.run()
