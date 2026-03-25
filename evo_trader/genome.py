"""
Genome: the DNA of a trading strategy.

Each genome is a dictionary of parameters that fully defines how an agent
trades. The genetic algorithm evolves these parameters over generations.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Dict, Any

# Bounds for each gene (min, max)
GENE_BOUNDS: Dict[str, tuple] = {
    # Moving average crossover
    "ma_fast_period": (5, 50),
    "ma_slow_period": (20, 200),

    # RSI
    "rsi_period": (5, 30),
    "rsi_oversold": (15, 40),
    "rsi_overbought": (60, 85),

    # Bollinger Bands
    "bb_period": (10, 50),
    "bb_std_dev": (1.0, 3.0),

    # MACD
    "macd_fast": (8, 20),
    "macd_slow": (20, 40),
    "macd_signal_period": (5, 15),

    # ATR (volatility filter)
    "atr_period": (7, 28),
    "atr_volatility_threshold": (0.01, 0.06),  # min volatility to trade

    # Stochastic
    "stoch_k_period": (5, 21),
    "stoch_d_period": (3, 7),
    "stoch_oversold": (15, 30),
    "stoch_overbought": (70, 85),

    # VWAP
    "vwap_period": (10, 40),

    # Signal weights (how much each indicator matters)
    "weight_ma_cross": (0.0, 1.0),
    "weight_rsi": (0.0, 1.0),
    "weight_bb": (0.0, 1.0),
    "weight_macd": (0.0, 1.0),
    "weight_stoch": (0.0, 1.0),
    "weight_obv": (0.0, 1.0),
    "weight_vwap": (0.0, 1.0),

    # Risk management
    "stop_loss_pct": (0.01, 0.15),       # 1% - 15%
    "take_profit_pct": (0.02, 0.30),     # 2% - 30%
    "position_size_pct": (0.1, 1.0),     # 10% - 100% of capital
    "trailing_stop_pct": (0.0, 0.10),    # 0% = disabled, up to 10%

    # Signal threshold: combined score must exceed this to trigger a trade
    "signal_threshold": (0.3, 0.8),
}

INT_GENES = {
    "ma_fast_period", "ma_slow_period", "rsi_period",
    "rsi_oversold", "rsi_overbought", "bb_period",
    "macd_fast", "macd_slow", "macd_signal_period",
    "atr_period", "stoch_k_period", "stoch_d_period",
    "stoch_oversold", "stoch_overbought", "vwap_period",
}

# Constraints: (gene_a, gene_b) means gene_a must be < gene_b
ORDERING_CONSTRAINTS = [
    ("ma_fast_period", "ma_slow_period"),
    ("rsi_oversold", "rsi_overbought"),
    ("macd_fast", "macd_slow"),
    ("stoch_oversold", "stoch_overbought"),
]


def random_genome() -> Dict[str, float]:
    """Create a completely random genome."""
    genome = {}
    for gene, (lo, hi) in GENE_BOUNDS.items():
        val = random.uniform(lo, hi)
        if gene in INT_GENES:
            val = int(round(val))
        genome[gene] = val

    return _clamp(genome)


def crossover(parent_a: Dict[str, float], parent_b: Dict[str, float]) -> Dict[str, float]:
    """Uniform crossover: each gene randomly comes from one parent."""
    child = {}
    for gene in GENE_BOUNDS:
        child[gene] = parent_a[gene] if random.random() < 0.5 else parent_b[gene]

    # Also do blend crossover for continuous genes with 30% probability
    for gene in GENE_BOUNDS:
        if gene not in INT_GENES and random.random() < 0.3:
            alpha = random.uniform(0.0, 1.0)
            child[gene] = alpha * parent_a[gene] + (1 - alpha) * parent_b[gene]

    return _clamp(child)


def mutate(genome: Dict[str, float], mutation_rate: float = 0.15) -> Dict[str, float]:
    """Mutate a genome by perturbing random genes."""
    genome = copy.deepcopy(genome)
    for gene, (lo, hi) in GENE_BOUNDS.items():
        if random.random() < mutation_rate:
            spread = (hi - lo) * 0.2  # mutate within 20% of range
            genome[gene] += random.gauss(0, spread)
            if gene in INT_GENES:
                genome[gene] = int(round(genome[gene]))
    return _clamp(genome)


def _clamp(genome: Dict[str, float]) -> Dict[str, float]:
    """Clamp all genes to their valid bounds and enforce constraints."""
    for gene, (lo, hi) in GENE_BOUNDS.items():
        genome[gene] = max(lo, min(hi, genome[gene]))
        if gene in INT_GENES:
            genome[gene] = int(round(genome[gene]))

    # Enforce ordering constraints
    for lo_gene, hi_gene in ORDERING_CONSTRAINTS:
        if genome[lo_gene] >= genome[hi_gene]:
            genome[hi_gene] = genome[lo_gene] + 1

    return genome


def genome_summary(genome: Dict[str, float]) -> str:
    """Human-readable summary of a genome."""
    lines = []
    lines.append(f"  MA Crossover: fast={genome['ma_fast_period']}, slow={genome['ma_slow_period']}")
    lines.append(f"  RSI: period={genome['rsi_period']}, oversold={genome['rsi_oversold']}, overbought={genome['rsi_overbought']}")
    lines.append(f"  Bollinger: period={genome['bb_period']}, std={genome['bb_std_dev']:.2f}")
    lines.append(f"  MACD: fast={genome['macd_fast']}, slow={genome['macd_slow']}, signal={genome['macd_signal_period']}")
    lines.append(f"  Stochastic: K={genome['stoch_k_period']}, D={genome['stoch_d_period']}, "
                 f"OS={genome['stoch_oversold']}, OB={genome['stoch_overbought']}")
    lines.append(f"  ATR: period={genome['atr_period']}, vol_thresh={genome['atr_volatility_threshold']:.3f}")
    lines.append(f"  VWAP: period={genome['vwap_period']}")
    lines.append(f"  Weights: MA={genome['weight_ma_cross']:.2f}, RSI={genome['weight_rsi']:.2f}, "
                 f"BB={genome['weight_bb']:.2f}, MACD={genome['weight_macd']:.2f}, "
                 f"Stoch={genome['weight_stoch']:.2f}, OBV={genome['weight_obv']:.2f}, "
                 f"VWAP={genome['weight_vwap']:.2f}")
    lines.append(f"  Risk: SL={genome['stop_loss_pct']:.1%}, TP={genome['take_profit_pct']:.1%}, "
                 f"size={genome['position_size_pct']:.1%}, trailing={genome['trailing_stop_pct']:.1%}")
    lines.append(f"  Signal threshold: {genome['signal_threshold']:.2f}")
    return "\n".join(lines)
