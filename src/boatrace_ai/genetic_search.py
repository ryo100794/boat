from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import random
import statistics
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar


CandidateT = TypeVar("CandidateT")


@dataclass(frozen=True)
class GeneticSearchSettings:
    population_size: int = 12
    generations: int = 5
    elite_count: int = 3
    mutation_rate: float = 0.30
    random_injections: int = 1
    max_workers: int = 4
    execution_backend: str = "thread"
    seed: int = 33034

    def validate(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least four")
        if not 1 <= self.elite_count <= self.population_size // 2:
            raise ValueError("elite_count must be between one and half the population")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between zero and one")
        if not 0 <= self.random_injections < self.population_size:
            raise ValueError("random_injections must be smaller than population")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.execution_backend not in {"thread", "process"}:
            raise ValueError("execution_backend must be thread or process")


@dataclass(frozen=True)
class EvaluatedCandidate(Generic[CandidateT]):
    candidate: CandidateT
    metrics: Mapping[str, Any]
    fitness: float
    first_generation: int


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min_fitness": ordered[0],
        "q1_fitness": percentile(0.25),
        "median_fitness": percentile(0.50),
        "q3_fitness": percentile(0.75),
        "max_fitness": ordered[-1],
        "std_fitness": statistics.pstdev(ordered),
    }


def evolve_population(
    *,
    settings: GeneticSearchSettings,
    evaluator: Callable[[CandidateT], Mapping[str, Any]],
    fitness: Callable[[Mapping[str, Any], CandidateT], float],
    random_candidate: Callable[[random.Random], CandidateT],
    crossover: Callable[[CandidateT, CandidateT, random.Random], CandidateT],
    mutate: Callable[[CandidateT, random.Random, float], CandidateT],
    candidate_key: Callable[[CandidateT], str],
    serialize: Callable[[CandidateT], Mapping[str, Any]],
    immigrants: Sequence[CandidateT] = (),
) -> tuple[list[EvaluatedCandidate[CandidateT]], list[dict[str, Any]]]:
    """Run a deterministic GA with cached fitness and stagnation recovery."""
    settings.validate()
    rng = random.Random(settings.seed)
    population = list(immigrants[: settings.elite_count])
    population.extend(
        random_candidate(rng)
        for _ in range(settings.population_size - len(population))
    )
    cache: dict[str, EvaluatedCandidate[CandidateT]] = {}
    history: list[dict[str, Any]] = []
    best_seen = -math.inf
    stagnant_generations = 0
    latest_ranked: list[EvaluatedCandidate[CandidateT]] = []

    for generation in range(settings.generations):
        unique = {candidate_key(row): row for row in population}
        attempts = 0
        while len(unique) < settings.population_size:
            candidate = random_candidate(rng)
            unique[candidate_key(candidate)] = candidate
            attempts += 1
            if attempts > settings.population_size * 100:
                raise ValueError("genetic candidate space lacks sufficient diversity")
        candidates = list(unique.values())[: settings.population_size]
        uncached = [row for row in candidates if candidate_key(row) not in cache]
        if uncached:
            workers = min(settings.max_workers, len(uncached))
            if settings.execution_backend == "process" and workers > 1:
                from joblib import Parallel, delayed

                metrics_rows = Parallel(
                    n_jobs=workers,
                    backend="loky",
                    batch_size="auto",
                )(delayed(evaluator)(candidate) for candidate in uncached)
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="genetic-fitness",
                ) as executor:
                    metrics_rows = list(executor.map(evaluator, uncached))
            for candidate, metrics in zip(uncached, metrics_rows):
                key = candidate_key(candidate)
                cache[key] = EvaluatedCandidate(
                    candidate=candidate,
                    metrics=dict(metrics),
                    fitness=float(fitness(metrics, candidate)),
                    first_generation=generation,
                )
        latest_ranked = sorted(
            (cache[candidate_key(row)] for row in candidates),
            key=lambda row: row.fitness,
            reverse=True,
        )
        current_best = latest_ranked[0].fitness
        if current_best > best_seen + 1e-9:
            best_seen = current_best
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        adaptive_rate = min(
            0.90, settings.mutation_rate + 0.15 * stagnant_generations
        )
        injection_count = (
            min(
                settings.population_size - settings.elite_count,
                settings.random_injections + stagnant_generations,
            )
            if generation + 1 < settings.generations
            else 0
        )
        history.append({
            "generation": generation,
            "best_fitness": current_best,
            "best_candidate": dict(serialize(latest_ranked[0].candidate)),
            "mutation_rate": adaptive_rate,
            "random_injections": injection_count,
            "unique_candidates": len(unique),
            "new_evaluations": len(uncached),
            "cached_evaluations": len(candidates) - len(uncached),
            "stagnant_generations": stagnant_generations,
            **_distribution([row.fitness for row in latest_ranked]),
        })

        elites = [row.candidate for row in latest_ranked[: settings.elite_count]]
        parent_pool = [
            row.candidate
            for row in latest_ranked[: max(settings.elite_count, len(latest_ranked) // 2)]
        ]
        population = list(elites)
        offspring_target = settings.population_size - injection_count
        while len(population) < offspring_target:
            contenders = rng.sample(parent_pool, min(3, len(parent_pool)))
            left = max(
                contenders, key=lambda row: cache[candidate_key(row)].fitness
            )
            contenders = rng.sample(parent_pool, min(3, len(parent_pool)))
            right = max(
                contenders, key=lambda row: cache[candidate_key(row)].fitness
            )
            population.append(
                mutate(crossover(left, right, rng), rng, adaptive_rate)
            )
        population.extend(random_candidate(rng) for _ in range(injection_count))

    all_ranked = sorted(cache.values(), key=lambda row: row.fitness, reverse=True)
    return all_ranked, history


__all__ = [
    "EvaluatedCandidate",
    "GeneticSearchSettings",
    "evolve_population",
]
