"""
Genetic Algorithm engine for evolving trading strategies.

Supports four evolution modes:
  1. Standard: train on one split, validate on another
  2. Walk-forward: rolling window evaluation
  3. Multi-asset: train across multiple assets
  4. Anti-overfit: validation-in-the-loop + noise injection + ensemble
"""

import random
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .genome import random_genome, crossover, mutate, genome_summary
from .backtest import (
    evaluate_genome,
    evaluate_genome_walk_forward,
    evaluate_genome_multi_asset,
    evaluate_with_validation,
    fitness_score,
)


@dataclass
class Individual:
    genome: Dict[str, float]
    fitness: float = 0.0
    result: Optional[Dict] = None
    generation: int = 0


def create_population(size: int) -> List[Individual]:
    """Create a random initial population."""
    return [Individual(genome=random_genome()) for _ in range(size)]


def evaluate_population(
    population: List[Individual],
    train_data,  # DataFrame or Dict[str, DataFrame]
    generation: int,
    mode: str = "standard",
    initial_capital: float = 10000.0,
    wf_splits: int = 5,
    val_data=None,  # for anti_overfit mode
) -> List[Individual]:
    """Evaluate all individuals using the specified mode."""
    for ind in population:
        if mode == "anti_overfit" and val_data is not None:
            ind.result = evaluate_with_validation(
                ind.genome, train_data, val_data,
                train_weight=0.4, val_weight=0.6,
                use_noise=True, initial_capital=initial_capital,
            )
        elif mode == "walk_forward":
            ind.result = evaluate_genome_walk_forward(
                ind.genome, train_data, n_splits=wf_splits,
                initial_capital=initial_capital,
            )
        elif mode == "multi_asset":
            ind.result = evaluate_genome_multi_asset(
                ind.genome, train_data, initial_capital=initial_capital,
            )
        else:
            ind.result = evaluate_genome(
                ind.genome, train_data, initial_capital=initial_capital,
            )

        fitness_mode = "anti_overfit" if mode == "anti_overfit" else (
            "robust" if mode != "standard" else "standard"
        )
        ind.fitness = fitness_score(ind.result, mode=fitness_mode)
        ind.generation = generation
    return population


def tournament_select(population: List[Individual], k: int = 3) -> Individual:
    """Select one individual via tournament selection."""
    tournament = random.sample(population, min(k, len(population)))
    return max(tournament, key=lambda ind: ind.fitness)


def genome_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Euclidean distance between two genomes (normalized)."""
    from .genome import GENE_BOUNDS
    dist_sq = 0.0
    for gene, (lo, hi) in GENE_BOUNDS.items():
        span = hi - lo if hi != lo else 1.0
        diff = (a[gene] - b[gene]) / span
        dist_sq += diff ** 2
    return dist_sq ** 0.5


def select_diverse_ensemble(population: List[Individual], n: int = 5) -> List[Individual]:
    """
    Select top individuals that are also diverse from each other.
    This prevents the ensemble from being N copies of the same strategy.
    Uses greedy farthest-point selection from the top candidates.
    """
    # Start from a pool of top 3x candidates
    candidates = sorted(population, key=lambda x: x.fitness, reverse=True)[:n * 3]
    if len(candidates) <= n:
        return candidates[:n]

    # Start with the best
    ensemble = [candidates[0]]
    remaining = candidates[1:]

    while len(ensemble) < n and remaining:
        # Pick the candidate most different from all current ensemble members
        best_candidate = None
        best_min_dist = -1

        for c in remaining:
            min_dist = min(genome_distance(c.genome, e.genome) for e in ensemble)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c

        if best_candidate:
            ensemble.append(best_candidate)
            remaining.remove(best_candidate)

    return ensemble


def evolve(
    train_data,  # DataFrame or Dict[str, DataFrame]
    val_data=None,  # DataFrame or Dict[str, DataFrame]
    population_size: int = 30,
    generations: int = 50,
    elite_count: int = 3,
    mutation_rate: float = 0.15,
    initial_capital: float = 10000.0,
    kill_threshold: float = 0.0,
    mode: str = "standard",
    wf_splits: int = 5,
    ensemble_size: int = 5,
    verbose: bool = True,
) -> Tuple[Individual, List[Dict]]:
    """
    Run the full evolutionary loop.

    Modes:
    - "standard": simple train/test split
    - "walk_forward": rolling window evaluation (anti-overfit)
    - "multi_asset": train on multiple assets (robustness)
    - "anti_overfit": validation-in-the-loop + noise injection

    Returns:
        best: The best individual found (or ensemble leader)
        history: List of per-generation stats
    """
    population = create_population(population_size)
    best_ever: Optional[Individual] = None
    history: List[Dict] = []

    for gen in range(generations):
        # --- Evaluate ---
        population = evaluate_population(
            population, train_data, gen,
            mode=mode, initial_capital=initial_capital,
            wf_splits=wf_splits, val_data=val_data,
        )

        # --- Sort by fitness ---
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # --- Track best ---
        gen_best = population[0]
        if best_ever is None or gen_best.fitness > best_ever.fitness:
            best_ever = copy.deepcopy(gen_best)

        # --- Stats ---
        fitnesses = [ind.fitness for ind in population]
        returns = [ind.result["total_return"] for ind in population]
        alive = [ind for ind in population if ind.result["total_return"] >= kill_threshold]
        killed = len(population) - len(alive)

        consistency = gen_best.result.get("consistency", None)
        consistency_str = f", Consistency: {consistency:.0%}" if consistency is not None else ""

        stats = {
            "generation": gen,
            "best_fitness": gen_best.fitness,
            "best_return": gen_best.result["total_return"],
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "avg_return": sum(returns) / len(returns),
            "alive": len(alive),
            "killed": killed,
            "best_trades": gen_best.result["total_trades"],
            "best_sharpe": gen_best.result["sharpe_ratio"],
            "best_drawdown": gen_best.result["max_drawdown"],
        }
        if consistency is not None:
            stats["best_consistency"] = consistency
        history.append(stats)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{generations}")
            print(f"  Best return: {gen_best.result['total_return']:+.2%}")
            print(f"  Best fitness: {gen_best.fitness:.2f}")
            print(f"  Avg return:  {stats['avg_return']:+.2%}")
            print(f"  Alive: {len(alive)}/{len(population)} (killed {killed})")
            print(f"  Trades: {gen_best.result['total_trades']}, "
                  f"Sharpe: {gen_best.result['sharpe_ratio']:.2f}, "
                  f"MaxDD: {gen_best.result['max_drawdown']:.2%}"
                  f"{consistency_str}")

            # Show train/val split in anti_overfit mode
            if mode == "anti_overfit":
                train_ret = gen_best.result.get("train_return", 0)
                val_ret = gen_best.result.get("val_return", 0)
                gap = gen_best.result.get("overfit_gap", 0)
                spread = gen_best.result.get("noise_spread", 0)
                print(f"  Train: {train_ret:+.2%} | Val: {val_ret:+.2%} | "
                      f"Gap: {gap:+.2%} | Noise spread: {spread:.3f}")

            fees = gen_best.result.get("total_fees", 0)
            if fees > 0:
                print(f"  Fees paid: ${fees:,.2f}")

            per_asset = gen_best.result.get("per_asset", {})
            if per_asset:
                for ticker, info in per_asset.items():
                    print(f"    {ticker}: {info['return']:+.2%} "
                          f"(Sharpe: {info['sharpe']:.2f}, "
                          f"DD: {info['drawdown']:.2%}, "
                          f"Trades: {info['trades']})")

        # --- Selection & Breeding ---
        elites = [copy.deepcopy(ind) for ind in population[:elite_count]]
        survivors = alive if len(alive) >= 4 else population[:max(4, len(population) // 2)]

        new_population = list(elites)
        while len(new_population) < population_size:
            if len(survivors) >= 2 and random.random() < 0.7:
                parent_a = tournament_select(survivors)
                parent_b = tournament_select(survivors)
                child_genome = crossover(parent_a.genome, parent_b.genome)
            else:
                parent = tournament_select(survivors)
                child_genome = copy.deepcopy(parent.genome)

            child_genome = mutate(child_genome, mutation_rate)
            new_population.append(Individual(genome=child_genome))

        population = new_population

    # --- Build ensemble from final population ---
    # Re-evaluate final population to get current fitness
    population = evaluate_population(
        population, train_data, generations,
        mode=mode, initial_capital=initial_capital,
        wf_splits=wf_splits, val_data=val_data,
    )
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    ensemble = select_diverse_ensemble(population, ensemble_size)

    if verbose and len(ensemble) > 1:
        print(f"\n{'='*60}")
        print(f"ENSEMBLE ({len(ensemble)} diverse agents):")
        for i, ind in enumerate(ensemble):
            ret = ind.result["total_return"]
            val = ind.result.get("val_return", "N/A")
            val_str = f"{val:+.2%}" if isinstance(val, float) else val
            print(f"  Agent {i+1}: return={ret:+.2%}, val={val_str}, "
                  f"fitness={ind.fitness:.2f}")

    # Store ensemble genomes in best_ever result
    if best_ever is not None:
        best_ever.result["ensemble"] = [
            {"genome": ind.genome, "fitness": ind.fitness,
             "return": ind.result["total_return"]}
            for ind in ensemble
        ]

    # --- Final validation (held-out test) ---
    if val_data is not None and best_ever is not None and mode != "anti_overfit":
        # In anti_overfit mode, val was already used during training,
        # so we split off a final test set
        if mode == "multi_asset" and isinstance(val_data, dict):
            val_result = evaluate_genome_multi_asset(
                best_ever.genome, val_data, initial_capital
            )
        elif isinstance(val_data, pd.DataFrame):
            val_result = evaluate_genome(best_ever.genome, val_data, initial_capital)
        else:
            val_result = None

        if val_result:
            if verbose:
                print(f"\n{'='*60}")
                print("VALIDATION (out-of-sample):")
                print(f"  Return: {val_result['total_return']:+.2%}")
                print(f"  Sharpe: {val_result['sharpe_ratio']:.2f}")
                print(f"  MaxDD:  {val_result['max_drawdown']:.2%}")
                print(f"  Trades: {val_result['total_trades']}")
                print(f"  Fees:   ${val_result.get('total_fees', 0):,.2f}")

                per_asset = val_result.get("per_asset", {})
                if per_asset:
                    for ticker, info in per_asset.items():
                        print(f"    {ticker}: {info['return']:+.2%}")

                train_ret = best_ever.result["total_return"]
                val_ret = val_result["total_return"]
                if train_ret > 0 and val_ret < train_ret * 0.3:
                    print("  WARNING: Possible overfitting detected!")
                elif val_ret > 0:
                    print("  Strategy generalizes to unseen data!")

            best_ever.result["validation"] = val_result

    # For anti_overfit mode, the val performance is already in the result
    if mode == "anti_overfit" and best_ever is not None and verbose:
        print(f"\n{'='*60}")
        print("FINAL RESULT (anti-overfit mode):")
        print(f"  Train return:    {best_ever.result.get('train_return', 0):+.2%}")
        print(f"  Val return:      {best_ever.result.get('val_return', 0):+.2%}")
        print(f"  Overfit gap:     {best_ever.result.get('overfit_gap', 0):+.2%}")
        print(f"  Noise spread:    {best_ever.result.get('noise_spread', 0):.4f}")
        val_ret = best_ever.result.get('val_return', 0)
        if val_ret > 0:
            print("  Strategy generalizes to unseen data!")
        else:
            print("  WARNING: Strategy still loses on validation data")

    return best_ever, history
